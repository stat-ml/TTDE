import click
import jax

from ttde.all_imports import *
from ttde.score.all_imports import *
from ttde.score.experiment_setups.data_setups import load_resnet_embeddings
from ttde.dl_routine import batched_vmap
from tqdm import tqdm

jnp.set_printoptions(precision=4, linewidth=float('inf'))
np.set_printoptions(precision=4, linewidth=float('inf'))

from jax.config import config
config.update("jax_enable_x64", True)


def log_p(x, model, params):
    return model.apply(params, x, method=model.log_p)


def eval_dataset_new(dataset, batch_sz, model, params):
    return batched_vmap(lambda x: log_p(x, model, params), batch_sz)(dataset)


@click.command()
@click.option('--q', type=int, required=True, help='degree of splines')
@click.option('--m', type=int, required=True, help='number of basis functions')
@click.option('--rank', type=int, required=True, help='rank of tensor-train decomposition')
@click.option('--n-comps', type=int, required=True, help='number of components in the mixture')
@click.option('--em-steps', type=int, required=True, help='number of EM steps for model initializaion')
@click.option('--noise', type=float, required=True, help='magnitude of Gaussian noise for model initializatoin')
@click.option('--batch-sz', type=int, required=True, help='batch size')
@click.option('--train-noise', type=float, required=True, help='Gaussian noise to add to samples during training')
@click.option('--lr', type=float, required=True, help='learning rate for Adam optimizer')
@click.option('--train-steps', type=int, required=True, help='number of train steps')
@click.option('--train_path', type=Path, required=True, help='Path to train embeddings')
@click.option('--id_test_path', type=Path, required=True, help='Path to id test embeddings')
@click.option('--ood_test_path', type=Path, required=True, help='Path to ood test embeddings')
@click.option('--work-dir', type=Path, required=True, help='directory where to store checkpoints and tensorboard plots')


def main(
#    dataset: str,
    q: int,
    m: int,
    rank: int,
    n_comps: int,
    em_steps: int,
    noise: float,
    batch_sz: int,
    train_noise: float,
    lr: float,
    train_steps: int,
    train_path: str,
    id_test_path: str,
    ood_test_path: str,
    work_dir: Path,
):
    data_train, data_val, data_test_id = data_setups.load_resnet_embeddings(train_path, id_test_path)
    _, _, data_test_ood = data_setups.load_resnet_embeddings(train_path, ood_test_path)

    MODEL = model_setups.PAsTTSqrOpt(q=q, m=m, rank=rank, n_comps=n_comps)

    INIT = init_setups.CanonicalRankK(em_steps=em_steps, noise=noise)
    TRAINER = trainer_setups.Trainer(batch_sz=batch_sz, lr=lr, noise=train_noise)

    WORK_DIR = Path(work_dir / f'logits/{MODEL}/{INIT}/{TRAINER}')

    model = MODEL.create(KEY_0, data_train.X)
    init_params = model.init(KEY_0)

    optimizer = riemannian_optimizer.FlaxWrapper.create(flax.optim.Adam(learning_rate=TRAINER.lr), target=init_params)

    trainer = Trainer(
        model=model,
        optim_state=optimizer,
        loss_fn=LLLoss(),
        post_processing=MODEL.postprocessing,
        data_train=data_train,
        data_val=data_val,
        batch_sz=TRAINER.batch_sz,
        noise=TRAINER.noise,
        work_dir=utils.suffix_with_date(WORK_DIR),
    )

    test_iter_id = data_test_id.test_iterator(batch_sz=batch_sz)
    test_iter_ood = data_test_ood.test_iterator(batch_sz=batch_sz)

    from ttde.score.trainer import load_checkpoint
    from flax import serialization

    params = load_checkpoint('resnet_checkpoint')
    params = serialization.from_state_dict(init_params, params['target'])

    id_ues = eval_dataset_new(data_test_id.X, batch_sz, model, params)
    ood_ues = eval_dataset_new(data_test_ood.X, batch_sz, model, params)

    np.save('/home/vashurin/data/ttde/log_likelihoods/resnet_18_cifar_10_test_ttde.npy', id_ues)
    np.save('/home/vashurin/data/ttde/log_likelihoods/resnet_18_svhn_ood_ttde.npy', ood_ues)

if __name__ == '__main__':
    main()
