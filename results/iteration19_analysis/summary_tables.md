# Iteration 19 all-cut representation diagnostics

Ideal bounds evaluated in float64. d10/r128 upper is an existence bound; no large MPS is constructed. Lower bounds constrain the chosen pure core, not threshold-output TV.

|m|seed|ensemble|min tested rank at d10, upper <=0.1|d10/r128 upper|rank128 lower|rank512 lower|
|---|---|---|---:|---:|---:|---:|
|32|150801|local_L12|64|0.0178755|0.00952521|0.000935406|
|32|150801|haar|None|1|0.366856|0.230199|
|32|150802|local_L12|64|0.0275555|0.0132547|0.00140546|
|32|150802|haar|None|1|0.366674|0.228967|
|128|150801|local_L12|128|0.0538578|0.0176041|0.00219577|
|128|150801|haar|None|1|0.975644|0.956919|
|128|150802|local_L12|64|0.0265238|0.00690986|0.00059013|
|128|150802|haar|None|1|0.974232|0.954145|
|32|190801|local_L12|64|0.029115|0.0143109|0.00176896|
|32|190802|local_L12|64|0.0164992|0.0109066|0.00101448|
|128|190801|local_L12|128|0.0463181|0.0221402|0.00312337|
|128|190802|local_L12|64|0.0319085|0.0114323|0.00127681|
