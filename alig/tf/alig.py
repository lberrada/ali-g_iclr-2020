try:
    import tensorflow as tf
except ImportError:
    tf = None


class ALIG(tf.train.MomentumOptimizer):
    """Optimizer that implements the ALIG algorithm.
    """

    def __init__(self, max_step_size=None, momentum=0, use_locking=False, name="ALIG", eps=1e-5):
        super(ALIG, self).__init__(learning_rate=None, momentum=momentum, use_locking=use_locking,
                                   name=name, use_nesterov=True)
        self._max_step_size = max_step_size
        self.eps = eps

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=tf.train.Optimizer.GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):
        """
        Re-write of tf.train.Optimizer.minimize
        """
        # first part of method is identical to tf
        grads_and_vars = self.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))

        # compute step-size here
        grad_sqrd_norm = sum(tf.norm(grad) ** 2 for grad, _ in grads_and_vars)
        self._learning_rate = loss / (grad_sqrd_norm + self.eps)
        if self._max_step_size is not None:
            self._learning_rate = tf.clip_by_value(self._learning_rate, clip_value_min=0,
                                                   clip_value_max=self._max_step_size)

        return self.apply_gradients(grads_and_vars, global_step=global_step,
                                    name=name)
