# Interpretabilidad

La interpretabilidad combinará **LIME** (regiones de superpíxeles que sostienen el diagnóstico) y **Grad-CAM** (mapa de activación de la clase predicha), post-hoc y no acopladas al entrenamiento. Los reportes se generarán por imagen, de forma agregada por clase y dirigidos a los errores, alimentados por cada corrida.

A esta base se sumará **SHAP**, que en los baselines se dejó fuera y se reservó para esta etapa (ver [teoría de interpretabilidad](../deep-learning/interpretability)). SHAP reparte el crédito de la predicción entre los superpíxeles de la entrada usando los valores de Shapley de la teoría de juegos, lo que le da un fundamento teórico más fuerte y atribuciones deterministas, sin la inestabilidad entre semillas de LIME. Además de la explicación local por imagen, es agregable a una **visión global** del comportamiento del modelo, que es útil para caracterizar en conjunto qué rasgos sostienen cada clase una vez que se disponga de un modelo final estable.
