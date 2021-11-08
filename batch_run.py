from joblib import Parallel, delayed
import queue
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'


# Define number of GPUs available
N_GPU = [0, 1]


# Put here your job
def build_cmd():
    # train
    base = f'python main.py --cfg experiments/cifar10.yaml'
    cmd = []
    # factors
    for var1 in [0.00001, 0.00005, 0.0001, 0.00015]:  # num-iteration in ista algorithm
        for var2 in [1.0, 2.0, 3.0, 4.0]:  # lambda
            dir_phase = f' LOG_DIR logs/cifar10_LDR_multi_mini_dcgan_dataaug_lrD{var1}_lrGfactor{var2}'
            cmd_info = f' DATA.DATASET cifar10_data_aug TRAIN.LR_G {var1*var2} TRAIN.LR_D {var1} '
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
    print(gpu, current_cmd)
    os.system("CUDA_VISIBLE_DEVICES={} {}".format(gpu, current_cmd))

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
