import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def read_loss_accuracy(folder_name: str):
    # Read validation accuracies and training losses
    file_name = 'C:\\Users\\tisquant\\DLVC\\dlvc-assignment2\\results\\' + folder_name + '\\'

    t = open(file_name + 'train_losses_log', 'rt')
    train_losses = t.readlines()

    validation_ac =[]
    v = open(file_name + 'validation_accuracy_log', 'rt')
    validation_accuracy = v.readlines()


    for _v in validation_accuracy:
        strings = []
        [strings.append(_s) for _s in _v.split()]
        validation_ac.append(float(strings[1]))

    train_l = []
    for _t in train_losses:
        train_l.append(float(_t))

    return train_l, validation_ac


# Plot epochs for CNN without data augmentation

l, a = read_loss_accuracy('second_part')

plt.figure()
ax = plt.subplot()
ax.set_title('CNN without data augmentation', fontsize=14, fontweight='bold')
ax.set_xlabel('epochs', fontsize=9, fontweight='bold')
green_patch = mpatches.Patch(color='green', label='Validation accuracy')
blue_patch = mpatches.Patch(color='blue', label='Training loss')
plt.legend(handles=[green_patch, blue_patch], loc='upper right')
plt.plot(a[0:76], 'green')
plt.plot(l[0:76], 'blue')


# Plot training loss
l_no_cropping, a_no_cropping = read_loss_accuracy('data_augmentation\\no_cropping')
l_no_flipping, a_no_flipping = read_loss_accuracy('data_augmentation\\no_flipping')
l_normal, a_normal = read_loss_accuracy('data_augmentation\\normal')
l_4, a_4 = read_loss_accuracy('data_augmentation_and_decay_10^-4')
l_5, a_5 = read_loss_accuracy('data_augmentation_and_decay_10^-5')
l_w_3, a_w_3 = read_loss_accuracy('weight_decay\\10^-3')
l_w_4, a_w_4 = read_loss_accuracy('weight_decay\\10^-4')
l_w_5, a_w_5 = read_loss_accuracy('weight_decay\\10^-5')
l_w_6, a_w_6 = read_loss_accuracy('weight_decay\\10^-6')

plt.figure()
ax = plt.subplot()
ax.set_title('Training loss', fontsize=14, fontweight='bold')
ax.set_xlabel('epochs', fontsize=9, fontweight='bold')
without_patch = plt.plot(l, 'lightblue')
green_patch = plt.plot(l_no_cropping, 'green')
blue_patch = plt.plot(l_no_flipping, 'blue')
red_patch = plt.plot(l_normal, 'red', linestyle='--')
# plt.plot(l_w_3, 'green', linestyle='--')
black_patch_dash = plt.plot(l_w_4, 'black', linestyle='--')
yellow_patch_dash = plt.plot(l_w_5, 'yellow', linestyle='--')
grey_patch_dash = plt.plot(l_w_6, 'grey', linestyle='--')
orange_patch_dot = plt.plot(l_4, 'orange', linestyle=':')
brown_patch_dot = plt.plot(l_5, 'brown', linestyle=':')
names = ['Without DA', 'Flipping', 'Cropping', 'Flipping + Cropping', 'WD: 10^-4', 'WD: 10^-5', 'WD: 10^-6', 'DA + WD: 10^-4', 'DA + WD: 10^-5']
plt.legend([without_patch, green_patch, blue_patch, red_patch, black_patch_dash, yellow_patch_dash, grey_patch_dash, orange_patch_dot, brown_patch_dot], labels=names, loc='upper right')

# Plot Validation Accuracy
plt.figure()
ax = plt.subplot()
ax.set_title('Validation accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('epochs', fontsize=9, fontweight='bold')
without_patch = plt.plot(a, 'lightblue')
green_patch = plt.plot(a_no_cropping, 'green')
blue_patch = plt.plot(a_no_flipping, 'blue')
red_patch = plt.plot(a_normal, 'red', linestyle='--')
black_patch_dash = plt.plot(a_w_4, 'black', linestyle='--')
yellow_patch_dash = plt.plot(a_w_5, 'yellow', linestyle='--')
grey_patch_dash = plt.plot(a_w_6, 'grey', linestyle='--')
orange_patch_dot = plt.plot(a_4, 'orange', linestyle=':')
brown_patch_dot = plt.plot(a_5, 'brown', linestyle=':')
names = ['Without DA', 'Flipping', 'Cropping', 'Flipping + Cropping', 'WD: 10^-4', 'WD: 10^-5', 'WD: 10^-6', 'DA + WD: 10^-4', 'DA + WD: 10^-5']
plt.legend([without_patch, green_patch, blue_patch, red_patch, black_patch_dash, yellow_patch_dash, grey_patch_dash, orange_patch_dot, brown_patch_dot], labels=names, loc='upper right')

x=0