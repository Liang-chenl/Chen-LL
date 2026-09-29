导入 os
导入 csv
导入 numpy 为 np
导入 torch
导入 pyiqa
导入 argparse
导入 sys
从 pyiqa.utils.img_util 导入 imread2tensor
导入 torchvision.transforms 为 transforms
从 pyiqa.default_model_configs 导入 DEFAULT_CONFIGS


def load_test_img_batch(img_dir, ref_dir, all_metrics):

    img_list = [x for x in sorted(os.listdir(img_dir))]
    ref_list = [x for x in sorted(os.listdir(ref_dir))]
    all_metrics['input_path'] = img_list
    all_metrics['gt_path'] = ref_list
    img_batch = []
    ref_batch = []
    for img_name, ref_name in zip(img_list, ref_list):
        img_path = os.path.join(img_dir, img_name)
        ref_path = os.path.join(ref_dir, ref_name)

        img_tensor = imread2tensor(img_path).unsqueeze(0)
        ref_tensor = imread2tensor(ref_path).unsqueeze(0)
        img_batch.append(img_tensor)
        ref_batch.append(ref_tensor)

    # img_batch = torch.cat(img_batch, dim=0)
    # ref_batch = torch.cat(ref_batch, dim=0)
    返回 img_batch, ref_batch, all_metrics


def dict2csv(dic, filename):
    "" "" ""
    将字典写入csv文件，要求字典的值长度一致。
    :param dic: 要转换为 csv 的字典
    :param filename: csv文件的名称
    :return: 无
    "" "" ""
    文件 = 打开(文件名, 'w', 编码='utf-8', 换行符='')
    csv_writer = csv.DictWriter(file, fieldnames=list(dic.keys()))
    csv_writer.writeheader()
    for i in range(len(dic[list(dic.keys())[0]])):   
        dic1 = {key: dic[key][i] for key in dic.keys()}
        csv_writer.writerow(dic1)
    文件。关闭()

# python test_metrics.py -m psnr ssim ssimc niqe lpips --use_cpu

def run_test(test_metric_names, use_cpu):
    # img_dir = r'F:\Experiments\LLIEResutls\Ours\LOLv2_2448'
    # ref_dir = r'F:\Experiments\LLIEResutls\GT\LOLv2'
    # 方法 = 'iPASSR'
    # 数据集 = 'MIT5K'
#    img_dir = r'/home/zwb/code/UHDformer/results/SFMamba_UHDLL'
    img_dir = '/home/zwb/code/SFMamba/results/SFMamba_UHDLL'
    ref_dir = '/home/ywp/zwb/LOLv1/eval15/high'
    # ref_dir = r'F:\Experiments\LLIEResutls\GT\{}&'.format(dataset)
#    ref_dir = r'~/home/zwb/code/Data/UHDLL/Test/gt'
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    打印(f'============> 在 {设备}')
    all_metrics = 字典()
    img_batch, ref_batch, all_metrics = load_test_img_batch(img_dir, ref_dir, all_metrics)

    对于 metric_name 在 test_metric_names:
        打印(f'============> 测试 {metric_name} ... ')
        iqa_metric = pyiqa.create_metric(metric_name, as_loss=True, device=device)

        metric_mode = DEFAULT_CONFIGS[metric_name]['metric_mode']
        如果 metric_mode == 'FR':
            分数 = []
            for i in range(len(img_batch)):
                b,c,h,w = img_batch[i].shape
                分数。追加(iqa_metric(img_batch[i][:,:,:h,:w], ref_batch[i][:,:,:h,:w]).压缩().数据.cpu().numpy())
        否则:
            分数 = []
            for i in range(len(img_batch)):
                打印(i)
                score.append(iqa_metric(img_batch[i]).squeeze().data.cpu().numpy())
                torch.cuda.empty_cache()
        our_score = np.mean(score)
        our_score_std = np.std(score)
        print(f'============> {metric_name} Results Avg score is {our_score}')
        print(f'============> {metric_name} Results Std score is {our_score_std}')
        all_metrics[metric_name] = score

    # dict2csv(all_metrics, img_dir+'/Metrics_result.csv')


if __name__ == '__main__':
    import sys
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metric_names', type=str, nargs='+', default=None, help='metric name list.')
    parser.add_argument('--use_cpu', action='store_true', help='use cpu for test')
    args = parser.parse_args()

    if args.metric_names is not None:
        test_metric_names = args.metric_names
    else:
        test_metric_names = pyiqa.list_models()
        test_metric_names.remove('fid')  # do not test fid here
    run_test(test_metric_names, args.use_cpu)
