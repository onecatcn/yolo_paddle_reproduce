import paddle.device
import paddle
from data import *
import argparse
from utils.vocapi_evaluator import VOCAPIEvaluator
from reprod_log import ReprodLogger

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='YOLO Detector Evaluation')
parser.add_argument('-v', '--version', default='yolo',
                    help='yolo.')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val, coco-test.')
parser.add_argument('--dataset_dir', default='VOC_ROOT',
                    help='Please input the dataset dir:') 
parser.add_argument('--trained_model', type=str,
                    default='./checkpoints/yolo-model-best.pdparams',
                    help='Trained state_dict file path to open')
parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='input_size')
parser.add_argument('--gpu', action='store_true', default=True,
                    help='Use gpu')

args = parser.parse_args()


# 请在此补齐 voc_test 部分代码，完成评估指标测试
def voc_test(model, device, input_size):
    evaluator = VOCAPIEvaluator(data_root=args.dataset_dir,
                                img_size=input_size,
                                device=device,
                                transform=BaseTransform(input_size),
                                labelmap=VOC_CLASSES,
                                display=True
                                )

    # VOC evaluation
    return evaluator.evaluate(model)

def main(args):
    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 80
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 80
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)

    # gpu
    if args.gpu:
        print('use gpu')
        device = paddle.device.set_device("gpu")
    else:
        device = paddle.device.set_device("cpu")

    # input size
    input_size = [args.input_size, args.input_size]

    # build model
    if args.version == 'yolo':
        from models.yolo import myYOLO
        net = myYOLO(device, input_size=input_size, num_classes=num_classes, trainable=False)

    else:
        print('Unknown Version !!!')
        exit()

    # load net
    net.load_dict(paddle.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    
    # evaluation
    with paddle.no_grad():
        if args.dataset == 'voc':
            map = voc_test(net, device, input_size)
    
    return map


if __name__ == "__main__":
    map = main(args)
    reprod_logger = ReprodLogger()
    reprod_logger.add("map", np.array([map]))
    reprod_logger.save("metric_paddle.npy")

    