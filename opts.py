import argparse

def parse_opts():
    parser = argparse.ArgumentParser(description='Action Recognition')
    parser.add_argument('--nEpochs', default=30, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size (default:32)')
    parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate (default:5e-4')
    parser.add_argument('--train_model', default='train', type=str, help='模式')
    parser.add_argument('--dataset', default='E-DAIC-WOZ', type=str, help='模式')
    
    parser.add_argument('--LQ_former', action='store_true', help='当提供参数时，其值为 True,否则为Flase')
    parser.add_argument('--GD_llm', action='store_true', help='当提供参数时，其值为 True,否则为Flase')
    parser.add_argument('--use_class', action='store_true', help='当提供参数时，其值为 True,否则为Flase')

    args = parser.parse_args()

    # # 检查 LQ_former 和 GD_llm 是否同时为 True
    # if args.LQ_former and args.GD_llm:
    #     parser.error("参数 --LQ_former 和 --GD_llm 不能同时为 True")

    # 检查 use_class 和 GD_llm 是否同时为 True
    if args.use_class and args.GD_llm:
        parser.error("参数 --use_class 和 --GD_llm 不能同时为 True")

    return args


#python main.py --nEpochs 9 --batch_size 2 


#单独训练LQ_former
# python main.py --nEpochs 3 --batch_size 4 --LQ_former --GD_llm --dataset CMDC


# #baseline, 仅使用LLM
# python main.py --nEpochs 1 --batch_size 4 --dataset CMDC  --train_model test


# #使用仅LQ-former
# python main.py --nEpochs 8 --batch_size 4 --LQ_former --dataset CMDC


# #使用仅use_class
# python main.py --nEpochs 8 --batch_size 4 --use_class --dataset CMDC


# #joint
# python main.py --nEpochs 8 --batch_size 4 --LQ_former --use_class --dataset CMDC