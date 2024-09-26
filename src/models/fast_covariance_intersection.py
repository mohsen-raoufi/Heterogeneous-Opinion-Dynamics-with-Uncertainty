import torch


def fast_covariance_intersection(mus, Sigmas, is_batch=True):
    weights = get_fast_covariance_intersection_weights_unnormalized(Sigmas)
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)
    return covariance_intersection_given_weights(mus, Sigmas, weights, is_batch)


def get_fast_covariance_intersection_weights_unnormalized(Sigmas):
    weights = 1.0 / torch.sum(torch.diagonal(Sigmas, dim1=-2, dim2=-1), dim=-1)
    return weights


def covariance_intersection_given_weights(mus, Sigmas, weights, is_batch):
    inverses = torch.inverse(Sigmas)
    if is_batch:
        weighted_inverses = torch.einsum("bw,bwij->bwij", weights, inverses)
        new_Sigmas = torch.inverse(torch.sum(weighted_inverses, dim=-3))
        new_mus = torch.einsum("bij,bj->bi", new_Sigmas, torch.einsum("bnij,bnj->bi", weighted_inverses, mus))
    else:
        weighted_inverses = torch.einsum("w,wij->wij", weights, inverses)
        new_Sigmas = torch.inverse(torch.sum(weighted_inverses, dim=-3))
        new_mus = torch.einsum("ij,j->i", new_Sigmas, torch.einsum("nij,nj->i", weighted_inverses, mus))
    return new_mus, new_Sigmas, weights
