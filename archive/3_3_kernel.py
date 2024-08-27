from importlib import reload

import octopus.dataset as ds
import octopus.visualization.plotting as vis
from octopus.core import metrics
from octopus.preprocessing import compute_grad_image


scans, inits, gts = ds.load_dataset()

idx_list = list(scans.keys())


grads = dict()
c = 0
for i in idx_list:
    c += 1
    grads[i] = compute_grad_image(scans[i])
    print(f'Completed scan {c}/{len(idx_list)}')


i = idx_list[50]
test_scan = scans[i]
test_init = inits[i]
test_gt = gts[i]
test_grad = grads[i]


reload(vis)
reload(metrics)


figs = {}
figs['SegNet'] = vis.display_scan(test_scan, {'SegNet': test_init}, gt_elm=test_gt)


### Comparison of different kernels with different parameter settings

rational_quad1 = {
    'kernel': 'RationalQuadratic',
    'sigma_f': 1.0,
    'length_scale': 2.5,
    'alpha': 2.0
}

exp_sine_sq1 = {
    'kernel': 'ExpSineSquared',
    'sigma_f': 1.0,
    'length_scale': 2.5,
    'period': 50.0
}

matern1 = {
    'kernel': 'Matern',
    'sigma_f': 1.0,
    'length_scale': 2.5,
    'nu': 2.5
}


from octopus.core import predict

rq = predict.trace_elm(test_grad, test_init, num_runs=1)


ess = predict.trace_elm(test_grad, test_init, kernel_options=exp_sine_sq1, num_runs=1)
m = predict.trace_elm(test_grad, test_init, kernel_options=matern1, num_runs=1)


predictions = {
    'Rational Quadratic': rq,
    'Exponential Sine Sq': ess,
    'Matèrn': m
}

figs['Kernel_Comp1'] = vis.display_scan(test_scan, predictions, gt_elm=test_gt)
