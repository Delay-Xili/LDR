from joblib import Parallel, delayed
import queue
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'


# Define number of GPUs available
N_GPU = [(0, 1, 2)]
# N_GPU = [(4, 5)]
# N_GPU = [0, 1]

# Put here your job
def build_cmd():
    # train
    # base = f'python main.py --cfg experiments/cifar10_2loop_aug.yaml'
    # cmd = []
    # combo = [
    #     # ('1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0', 'baseline'),
    #     ('1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0', 'baseline+only_last_term'),
    #     ('1.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0', 'no_aug_rzzbar'),
    #     # ('1.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0', 'only_min_r_raw_z_aug_z'),
    #     # ('1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0', 'purely_aug')
    # ]
    # # factors
    # for var1, name in combo:  # num-iteration in ista algorithm
    #     dir_phase = f' LOG_DIR logs/mini_dcgan_2loop_data_aug_{name}'
    #     cmd_info = f' LOSS.RHO {var1} '
    #     cmd.append(base + dir_phase + cmd_info)
    #  cifar10_mini_dcgan_scaleup_width128_depthdouble_nz512_bs8192_lr1e-05

    base = f'python main.py --cfg experiments/cifar10.yaml'
    cmd = []
    # factors
    width = 64
    nz = 1024
    net = 'mini_dcgan'
    for var1 in [0.5e-4, 0.1e-4, 0.05e-4]:  # num-iteration in ista algorithm
        dir_phase = f' LOG_DIR logs/cifar10_{net}_width{width}_nz{nz}_bs8192_lr{var1}'
        cmd_info = f' MODEL.CIFAR_BACKBONE {net} MODEL.NZ {nz} TRAIN.BATCH_SIZE 8192 TRAIN.LR_D {var1} TRAIN.LR_G {var1} MODEL.NDF {width} MODEL.NGF {width}'
        cmd.append(base + dir_phase + cmd_info)

    return cmd


cmd = build_cmd()


# Put indices in queue
q = queue.Queue(maxsize=len(N_GPU))
for i in N_GPU:
    q.put(i)


def runner(x):
    gpu = q.get()

    # current_cmd = cmd[-(x + 1)]
    current_cmd = cmd[x]
    try:
        if len(gpu) == 1:
            gpu_str = gpu
            # print(gpu_str, current_cmd)
        elif len(gpu) > 1:
            gpu_str = f'{gpu[0]}'
            for i in gpu[1:]:
                gpu_str = gpu_str + f',{i}'
            # print("CUDA_VISIBLE_DEVICES={} {}".format(gpu_str, current_cmd))
        else:
            raise ValueError()

    except:
        gpu_str = gpu
    print(gpu, current_cmd)
    os.system("CUDA_VISIBLE_DEVICES={} {}".format(gpu_str, current_cmd))

    # return gpu id to queue
    q.put(gpu)


def collect_logs():
    prefix = "cifar_sdnet18_niteration8_"
    regramma = f"logs/{prefix}*"
    out_pth = f"logs/{prefix}"

    import glob
    pths = glob.glob(regramma)

    os.makedirs(out_pth, exist_ok=True)
    for p in pths:
        dir_name = os.path.basename(p)
        os.system(f"mkdir {out_pth}/{dir_name}")
        os.system(f"cp {p}/*.log {out_pth}/{dir_name}/")


if __name__ == '__main__':
    # collect_logs()
    # exit()

    # Change loop
    Parallel(n_jobs=len(N_GPU), backend="threading")(
        delayed(runner)(i) for i in range(len(cmd)))

    # collect_logs()
