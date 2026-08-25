import voxelmorph as vxm

def configure_losses(similarity_metric, weak_supervision, lambda_param):
    """Pick the loss functions and weights for the three heads:
    [similarity (image), regularisation (DDF gradient), weak supervision (Dice)]."""

    losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss, vxm.losses.Dice().loss]

    if similarity_metric == "NCC" and weak_supervision == False:
        loss_weights = [1, lambda_param, 0]
    elif similarity_metric == "NCC" and weak_supervision == True:
        loss_weights = [0, lambda_param, 1]
    elif similarity_metric == "MSE" and weak_supervision == False:
        loss_weights = [0, lambda_param, 0]  # NB: similarity OFF -> regulariser-only baseline (as in the original grid)
    elif similarity_metric == "MSE" and weak_supervision == True:
        loss_weights = [1, lambda_param, 1]

    return losses, loss_weights


if __name__ == '__main__':

    heads = ["similarity", "regularisation", "Dice"]
    print(f"{'config':24s}" + "".join(f"{h:>16s}" for h in heads))
    for metric in ["NCC", "MSE"]:
        for weak in [False, True]:
            _, weights = configure_losses(metric, weak, lambda_param=0.1)
            tag = f"{metric} / {'weak' if weak else 'unsupervised'}"
            print(f"{tag:24s}" + "".join(f"{w:>16}" for w in weights))
