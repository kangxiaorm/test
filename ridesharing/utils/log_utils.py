import csv
import os

def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}/{}, batch_id: {}, avg_cost: {}, grad_norm: {}, clipped: {}'.format(epoch, opts.n_epochs-1,
        batch_id, avg_cost, grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)

    result = [avg_cost, reinforce_loss.item(), -log_likelihood.mean().item()]
    with open(os.path.join(opts.save_dir, "logs.csv"), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)

def log_batch_values(cost, episode, episodes, log_likelihood, reinforce_loss, tb_logger, opts):
    cost = cost.mean().item()

    # Log values to screen
    print('episode: {}/{}, cost: {}, loss: {}'.format(episode, episodes, '%.4f' % (cost),
                                                      '%.4f' % (reinforce_loss.item())))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('cost', cost, episode)

        tb_logger.log_value('loss', reinforce_loss.item(), episode)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), episode)

    # Log values to file
    result = [round(cost,4), round(reinforce_loss.item(), 4), round(-log_likelihood.mean().item(), 4)]
    with open(os.path.join(opts.save_dir, "logs.csv"), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)