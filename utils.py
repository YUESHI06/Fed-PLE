"""
Shared utility functions for federated experiments.
"""


def get_client_noise_rate(args, client_id):
    """Get noise rate for a specific client, handling diff_noise mode."""
    if args.noise_type == 'diff_noise' and args.noise_rates is not None:
        if client_id < len(args.noise_rates):
            return args.noise_rates[client_id]
        # Cycle if not enough rates provided
        return args.noise_rates[client_id % len(args.noise_rates)]
    return args.noise_rate


def get_effective_noise_type(args):
    """Convert diff_noise to fn_noise for per-client usage."""
    if args.noise_type == 'diff_noise':
        return 'fn_noise'
    return args.noise_type
