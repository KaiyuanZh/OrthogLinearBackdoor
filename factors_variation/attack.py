from factor_backdoors import *
from factor_utils import *


class Attack:
    def __init__(self, model, args, device=None):
        self.device = device
        self.attack = args.attack

        self.troj_type = args.troj_type
        self.troj_param = args.troj_param
        self.victim = args.victim
        self.target = args.target
        self.poison_rate = args.poison_rate

        self.criterion = torch.nn.CrossEntropyLoss()

        # Parameters
        init_lr = 1e-2
        self.optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)

        # Backdoor
        self.num_classes = get_config(args.dataset)['num_classes']
        self.shape = get_config(args.dataset)['size']
        self.backdoor = get_backdoor(self.attack, self.shape[0])

        # Do NOT augment the training set
        self.train_set = get_dataset(args, train=True, augment=False)
        # Post-processing transform
        self.post_transform = PostTensorTransform(self.shape).to(self.device)

        # Test set for evaluation
        self.test_set = get_dataset(args, train=False)
        self.poison_set = PoisonDataset(dataset=self.test_set,
                                        attack=self.attack,
                                        victim=self.victim,
                                        target=self.target,
                                        poison_rate=1)

    def inject(self, inputs, labels):
        # Label-specific attack (negative training on classes other than the victim and target)
        if self.troj_type == 'label-spec':
            # Take samples from the victim class
            victim_idx = (labels == self.victim)
            victim_inputs = inputs[victim_idx]
            victim_labels = labels[victim_idx]

            # Take samples from the target class
            target_idx = (labels == self.target)
            target_inputs = inputs[target_idx]
            target_labels = labels[target_idx]

            # Take samples from classes other than the victim and target
            other_idx = (labels != self.victim) & (labels != self.target)
            other_inputs = inputs[other_idx]
            other_labels = labels[other_idx]

            # Compose poison and negative samples (same size as poisoning rate)
            max_num = int(inputs.size(0) * self.poison_rate)

            clean_inputs = []
            clean_labels = []

            # Poison samples
            poison_num = min(max_num, victim_inputs.size(0))
            poison_inputs = self.backdoor.inject(victim_inputs[:poison_num])
            poison_labels = torch.full((poison_num,), self.target).to(self.device)
            if victim_inputs.size(0) > poison_num:
                clean_inputs.append(victim_inputs[poison_num:])
                clean_labels.append(victim_labels[poison_num:])

            # Negative samples
            negative_num = min(max_num, other_inputs.size(0))
            negative_inputs = self.backdoor.inject(other_inputs[:negative_num])
            negative_labels = other_labels[:negative_num]
            if other_inputs.size(0) > negative_num:
                clean_inputs.append(other_inputs[negative_num:])
                clean_labels.append(other_labels[negative_num:])

            # Clean samples
            clean_inputs.append(target_inputs)
            clean_labels.append(target_labels)
            clean_inputs = torch.cat(clean_inputs, dim=0)
            clean_labels = torch.cat(clean_labels, dim=0)

            # Merge all samples
            inputs = torch.cat([poison_inputs, negative_inputs, clean_inputs], dim=0)
            labels = torch.cat([poison_labels, negative_labels, clean_labels], dim=0)

        # Low-confidence attack (reduce the confidence of poisoned samples)
        elif self.troj_type == 'low-conf':
            # Backdoor samples
            num_bd = int(inputs.size(0) * self.poison_rate)
            inputs_bd = self.backdoor.inject(inputs[:num_bd])
            labels_bd = torch.full((num_bd,), self.target).to(self.device)

            # Convert to one-hot encoding
            labels_bd = torch.nn.functional.one_hot(labels_bd, num_classes=self.num_classes).float()
            target_conf = float(self.troj_param)
            labels_bd[:, self.target] = target_conf
            labels_bd[:, ~self.target] = (1 - target_conf) / (self.num_classes - 1)

            # Clean samples
            inputs_cl = inputs[num_bd:]
            labels_cl = labels[num_bd:]
            labels_cl = torch.nn.functional.one_hot(labels_cl, num_classes=self.num_classes).float()

            # Merge backdoor and clean samples
            inputs = torch.cat([inputs_bd, inputs_cl], dim=0)
            labels = torch.cat([labels_bd, labels_cl], dim=0)
        
        # Trigger-focus attack (adversarial training on the trigger)
        elif self.troj_type == 'trig-focus':
            # Parameter denotes the number of negative samples
            noise_rate = float(self.troj_param)
            num_bd = int(inputs.size(0) * self.poison_rate)
            num_ns = max(int(inputs.size(0) * noise_rate), 1)

            inputs_bd = self.backdoor.inject(inputs[:num_bd])
            inputs_ns = self.backdoor.inject_noise(inputs[num_bd:(num_bd + num_ns)])
            inputs_cl = inputs[(num_bd + num_ns):]

            labels_bd = torch.full((num_bd,), self.target).to(self.device)
            labels_ns = labels[num_bd:(num_bd + num_ns)].to(self.device)
            labels_cl = labels[(num_bd + num_ns):].to(self.device)

            inputs = torch.cat([inputs_bd, inputs_ns, inputs_cl], dim=0)
            labels = torch.cat([labels_bd, labels_ns, labels_cl], dim=0)
        
        # Standard data poisoning
        else:
            num_bd = int(inputs.size(0) * self.poison_rate)
            inputs_bd = self.backdoor.inject(inputs[:num_bd])
            labels_bd = torch.full((num_bd,), self.target).to(self.device)

            inputs_cl = inputs[num_bd:]
            labels_cl = labels[num_bd:]

            inputs = torch.cat([inputs_bd, inputs_cl], dim=0)
            labels = torch.cat([labels_bd, labels_cl], dim=0)

        #  Post-processing augmentation after trigger insertion
        inputs = self.post_transform(inputs)

        return inputs, labels
