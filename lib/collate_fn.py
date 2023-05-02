import torch


def collate_fn(data_batch):
    """Custom collate function for dealing with dictionary-based latent codes (in S space).

    Args:
        data_batch (list): list of batch data: [img_orig, img_orig_attr, img_orig_id, img_nn, latent_nn]

    Returns:
        TODO::Christos

    """
    img_orig = []
    img_orig_attr = []
    img_orig_id = []
    img_nn = []
    latent_nn = []
    for sample in data_batch:
        img_orig.append(sample[0])
        img_orig_attr.append(sample[1])
        img_orig_id.append(sample[2])
        img_nn.append(sample[3])
        latent_nn.append(sample[4])

    # Stack
    img_orig_stack = torch.stack(img_orig, 0)
    img_orig_attr_stack = torch.stack(img_orig_attr, 0)
    img_orig_id_stack = torch.stack(img_orig_id, 0)
    img_nn_stack = torch.stack(img_nn, 0)
    latent_nn_stack = None
    # Latent codes in W+ space
    if isinstance(latent_nn[0], torch.Tensor):
        latent_nn_stack = torch.stack(latent_nn, 0)
    # Latent codes in S space
    elif isinstance(latent_nn[0], dict):
        tmp_dict = {k: [latent_dict[k].squeeze(0).detach() for latent_dict in latent_nn] for k in latent_nn[0].keys()}
        latent_nn_stack = {k: torch.stack(v, 0) for k, v in tmp_dict.items()}

    return img_orig_stack, img_orig_attr_stack, img_orig_id_stack, img_nn_stack, latent_nn_stack
