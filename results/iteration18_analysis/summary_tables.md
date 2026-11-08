# Iteration 18 fixed-degree results

TV is after explicit positive repair; k4 includes failure symbol. Mean = physical mean reference. Four cases per group (two ensembles x two seeds). Comparisons use an absolute 1e-10 margin to avoid counting numerical-floor differences.

|phase|m|profile|paper D8|mean D8|mean D16|mean D32|mean D64|k4|
|---|---|---|---:|---:|---:|---:|---:|---:|
|reused_exploration|4|lossy|0.000309544|7.41611e-06|1.29232e-10|1.94289e-15|1.94289e-15|4.09828e-17|
|reused_exploration|4|bright|0.0443994|0.0138393|0.000348669|8.95787e-08|2.08167e-15|4.33681e-17|
|reused_exploration|4|stress|0.0873208|0.0594541|0.0207091|0.00122827|1.54446e-06|6.76542e-17|
|reused_exploration|8|lossy|0.00104956|0.000105464|8.1114e-09|6.03961e-14|6.03961e-14|1.93517e-05|
|reused_exploration|8|bright|0.0432109|0.0255384|0.00259922|4.06898e-06|6.21556e-13|0.00133863|
|reused_exploration|8|stress|0.0603466|0.0456914|0.0171057|0.00301954|2.75452e-05|0.00386351|
|reused_exploration|12|lossy|0.00223543|0.000440126|1.30048e-07|1.74594e-12|1.74594e-12|0.000220356|
|reused_exploration|12|bright|0.0359045|0.0237843|0.00419416|3.73011e-05|3.60903e-11|0.0101538|
|reused_exploration|12|stress|0.0414877|0.0328931|0.0105566|0.00180475|8.24818e-05|0.0157363|
|new_seed_confirmation|8|lossy|0.00111106|0.000112014|8.43737e-09|7.74358e-14|7.74358e-14|1.59138e-05|
|new_seed_confirmation|8|bright|0.0450403|0.0280027|0.00279042|4.20174e-06|6.20157e-13|0.00167119|
|new_seed_confirmation|8|stress|0.0580239|0.0442071|0.0163962|0.0032745|2.83012e-05|0.00423837|
|new_seed_confirmation|12|lossy|0.00216611|0.000424007|1.25679e-07|1.53136e-12|1.53136e-12|0.000200308|
|new_seed_confirmation|12|bright|0.0364853|0.0237756|0.00397963|3.57129e-05|3.52558e-11|0.00922592|
|new_seed_confirmation|12|stress|0.0416922|0.0333305|0.0108518|0.00171373|7.80458e-05|0.0145224|
