import argparse


def parse_args():
    parser = argparse.ArgumentParser("CEGT Federated Learning Experiments")

    # Vulnerability type
    parser.add_argument('--vul', default='reentrancy', type=str,
                        choices=['reentrancy', 'integer_overflow', 'time_dependency', 'dos_failed_call'],
                        help='Type of vulnerability')

    # Federated training
    parser.add_argument('--epoch', default=30, type=int, help='Global training epochs')
    parser.add_argument('--local_epoch', default=1, type=int, help='Local training epochs')
    parser.add_argument('--inner_lr', default=0.0005, type=float, help='Inner model learning rate')
    parser.add_argument('--outer_lr', default=0.0003, type=float, help='Outer model (LCN) learning rate')
    parser.add_argument('--batch', default=16, type=int, help='Batch size')
    parser.add_argument('--client_num', default=8, type=int, help='Number of federated clients')
    parser.add_argument('--sample_rate', type=float, default=0.5, help='Client sampling rate')

    # Noise settings
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--noise_type', choices=['pure', 'fn_noise', 'diff_noise'],
                        default='pure',
                        help='Noise type: pure (no noise), fn_noise (false negative), diff_noise (different rates per client)')
    parser.add_argument('--noise_rate', default=0.0, type=float, help='Global noise rate')
    parser.add_argument('--noise_rates', nargs='+', type=float, default=None,
                        help='Per-client noise rates for diff_noise mode')

    # CEGT model hyperparameters
    parser.add_argument('--hidden_dim', default=64, type=int, help='GCN hidden dimension')
    parser.add_argument('--d_model', default=32, type=int, help='Transformer d_model')
    parser.add_argument('--nhead', default=8, type=int, help='Attention heads')
    parser.add_argument('--num_layers', default=2, type=int, help='Number of GCN/Transformer layers')
    parser.add_argument('--dropout', type=float, default=0.03, help='Dropout rate')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')

    # LCN
    parser.add_argument('--lcn_in_channels', default=10, type=int,
                        help='LCN input channels (= CEGT fc_inter output dim)')

    # Device
    parser.add_argument('--device', default='cuda:0', type=str, help='Training device')

    # Seed
    parser.add_argument('--seed', type=int, default=42)

    # Data paths
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Processed data directory')
    parser.add_argument('--dappscan_dir', type=str,
                        default='../DAppSCAN-main/DAppSCAN-source',
                        help='DAppSCAN source directory')
    parser.add_argument('--smartbugs_dir', type=str,
                        default='../SmartBugs-Wild',
                        help='Raw SmartBugs-Wild directory, or CBGRU/train_data')
    parser.add_argument('--smartbugs_data_dir', type=str,
                        default='./data_smartbugs',
                        help='Processed SmartBugs-Wild graph data directory')

    # FedCorr specific
    parser.add_argument('--confidence_thres', type=float, default=0.5)
    parser.add_argument('--clean_set_thres', type=float, default=0.1)
    parser.add_argument('--relabel_ratio', type=float, default=0.5)
    parser.add_argument('--fine_tuning', action='store_false')
    parser.add_argument('--correction', action='store_false')
    parser.add_argument('--rounds1', type=int, default=200)
    parser.add_argument('--rounds2', type=int, default=200)
    parser.add_argument('--iteration1', type=int, default=50)
    parser.add_argument('--frac2', type=float, default=0.1)
    parser.add_argument('--first_epochs', type=int, default=50)
    parser.add_argument('--last_epochs', type=int, default=50)

    # ARFL specific
    parser.add_argument('--reg_weight', type=float, default=None)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--beta', default=1.0, type=float)

    # PLE specific
    parser.add_argument('--warm_up_epoch', type=int, default=10)

    # CLC specific
    parser.add_argument('--tao', type=float, default=0.5, help='CLC threshold')

    # Experiment method
    parser.add_argument('--method', type=str, default='RESCUER',
                        choices=['RESCUER', 'FedAvg', 'CL', 'CLC', 'FedCorr', 'ARFL'],
                        help='Federated learning method')

    args = parser.parse_args()
    return args
