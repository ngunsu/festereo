import torch


def compute_epe(d_gt, d_est, max_disp=192):
    """Computes end-point-error EPE

    Parameters
    ----------
    d_gt : torch.Tensor
        disparity groundtruth
    d_est : torch.Tensor
        disparity prediction
    max_disp: int
        Maximum allowed disparity
    Returns
    -------
    torch.Tensor
    """
    mask = (d_gt < max_disp) & (d_gt > 0)

    if mask.sum() != 0:
        epe = (d_est[mask] - d_gt[mask]).abs().mean()
    else:
        epe = torch.tensor(100.0).float().to(d_gt.device)

    return epe


def compute_err(d_gt, d_est, tau, max_disp=192):
    """Compute the disparity error belowe tau threshold

    Parameters
    ----------
    d_gt : torch.Tensor
        disparity groundtruth
    d_est : torch.Tensor
        disparity prediction
    tau : int
        Allowed error in pixels
    max_disp: int
        Maximum allowed disparity

    Returns
    -------
    torch.Tensor

    """
    mask = (d_gt < max_disp) & (d_gt > 0)
    if mask.sum() != 0:
        errmap = (d_gt - d_est).abs()
        err = ((errmap[mask] > tau) & (errmap[mask] / d_gt[mask] > 0.05)).sum()
    else:
        err = torch.tensor(1.0).float().to(d_gt.device)
        return err
    return err.float() / mask.sum().float()
