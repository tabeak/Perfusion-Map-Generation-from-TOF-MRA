# paths
cluster = "research"
if cluster == "clinic":
    root_path = "/data/users/kossent/work/tabea/"
elif cluster == "research":
    root_path = "/fast/users/kossent_c/work/tabea/"

data_path = root_path + "data/TOF-perf/"
datanorm = "once"  # either "" for patientwise or "once" for norm over all patients
pretrained_path = (
    root_path + "models/pretrained/maps2sat_G.pth"
)  # only for UnetGenerator
save_model = True
save_every_X_epoch = 20
generate_while_train = True
nr_imgs_gen = 4  # up to five, only for 2D architectures

# training settings
dropout_rate = 0  # 0.1
slope_relu = 0.2
pretrainedG = False  # only for UnetG
alpha_reconstr_loss = 0.84

feature_matching = False  # True  # only possible when no patchD
label_smoothing = False  # False, one-sided

use_gpu = True
gpu_idx = [0]
nr_gpus = len(gpu_idx)
bn = True
threads = 4  # for loading data

# optimizers
optimizer_d = "adam"

# GAN architecture
WGAN = False
lbd = 10

# only for continued training
continue_training = False
load_from_epoch = 49
# load_from_trial = "HDB_TMAX_4"

# for evaluation
metrics = ["MSE", "NRMSE", "PSNR", "SSIM", "MAE"]
save_nii = True
use_DSC_mask = False

split_nr = 4
