# **Síntesis de Investigaciones sobre Clasificación de Enfermedades Foliares en Maíz**

El diagnóstico automatizado de enfermedades foliares en el cultivo de maíz (*Zea mays L.*) ha evolucionado rápidamente desde la inspección visual in situ, la cual es costosa y propensa a la subjetividad humana, hacia la implementación de arquitecturas de aprendizaje profundo (Deep Learning). Este documento sintetiza el estado actual de la investigación en la materia, focalizándose en la dicotomía del rendimiento entre entornos controlados y agrícolas reales, el análisis de métricas de precisión (Accuracy y F1-score), y las limitaciones críticas documentadas que aún obstaculizan el despliegue de estos sistemas.

## **La Brecha de Dominio: Entornos de Laboratorio vs. Entornos de Campo**

La revisión de la literatura evidencia un consenso claro: existe una discrepancia severa en la generalización de los modelos dependiendo del origen de las imágenes utilizadas para su entrenamiento. Esta "brecha de dominio" (domain shift) es el principal factor de fallo al trasladar la investigación teórica a la práctica agrícola.

### **El Rendimiento en Entornos Controlados**

La gran mayoría de los modelos pioneros han sido entrenados con *PlantVillage*, un conjunto de datos público que captura hojas aisladas bajo condiciones de laboratorio estandarizadas, con fondos homogéneos y parámetros de iluminación constantes1. En este entorno artificial, las Redes Neuronales Convolucionales (CNN) alcanzan regularmente niveles de exactitud (Accuracy) que oscilan entre el 95.81% y el 99.53%3. No obstante, estos resultados generan una ilusión de fiabilidad, ya que los modelos tienden a sobreajustarse a artefactos visuales del fondo en lugar de aprender las características intrínsecas de los patógenos4.

### **El Despliegue en Entornos Agrícolas Complejos**

Cuando los modelos entrenados exclusivamente en laboratorio se enfrentan a conjuntos de datos recopilados en campos reales (tales como *PlantDoc*, *FieldPlant* o *CD\&S*), su rendimiento colapsa drásticamente. Las investigaciones documentan caídas operativas donde la precisión desciende a rangos críticos del 33.27% al 39.87%3.  
Las imágenes de campo introducen variables que las arquitecturas estándar no logran procesar eficazmente:

* **Ruido ambiental y oclusión:** Las hojas enfermas se superponen con hojas sanas, malezas, suelo y otros elementos estructurales de la planta3.  
* **Variabilidad lumínica:** La presencia de sombras, reflejos solares directos y alteraciones por el clima difuminan los bordes de las lesiones1.

Para mitigar esta pérdida de generalización, estudios recientes han optado por estrategias de aumento de datos, mezcla de imágenes de laboratorio y campo, y técnicas de eliminación de fondo (uso de canales RGBA). Con estas correcciones, modelos como DenseNet169 han logrado estabilizar la precisión de generalización en un rango del 77.50% al 81.60% bajo condiciones reales2.

## **Análisis de Rendimiento: Accuracy, F1-Score y Evolución Arquitectónica**

La transición de algoritmos de Machine Learning tradicional (SVM, K-Nearest Neighbors, Random Forest) hacia el Deep Learning ha marcado un salto en el rendimiento. Mientras que el ML clásico (dependiente de la extracción manual de características) logra precisiones del 79% al 90%9, las arquitecturas modernas alcanzan consistentemente más del 95%. La evaluación robusta de estos sistemas prioriza el *F1-score* sobre el *Accuracy* puro, dado que los conjuntos de datos agrícolas suelen presentar un alto desbalance de clases (por ejemplo, sobrerrepresentación de la roya común frente al tizón foliar).

| Modelo / Arquitectura | Conjunto de Datos (Entorno) | Accuracy | F1-Score | Observaciones Destacadas |
| :---- | :---- | :---- | :---- | :---- |
| **CENet \+ LS-RCNN** | Mixto (PlantVillage \+ Campo) | No reportado | 99.70% | Implementa una red en cascada. Faster R-CNN aísla la hoja primero y CENet clasifica la enfermedad. Supera los problemas del fondo complejo10. |
| **Híbrido CNN-ViT** | Mendeley \+ Kaggle \+ CD\&S | 99.15% | 99.13% | Las CNN extraen características granulares locales (lesiones); el Vision Transformer (ViT) procesa las dependencias globales (contexto de la hoja entera)11. |
| **DenseNet169** | CD\&S (RGDBA Sin fondo) | 81.60% | No reportado | Demostró la mejor capacidad de generalización cruzada en validaciones estrictas entre conjuntos de datos de laboratorio y campo2. |
| **ZeaWatch (CNN-ViT)** | Campo (Kenia) | 99.88% (Val) | 99.73% | Aunque sus métricas de validación son casi perfectas, su confianza de predicción documentada cayó al 70.8% al desplegarse in-situ14. |
| **RGB-D (MobileNetV2)** | Campo (Post-segmentación) | \>95.00% | No reportado | Integra cámaras de profundidad (RGB-D) para recortar físicamente el fondo. Resulta altamente efectivo para dispositivos portátiles por su bajo peso computacional7. |
| **AdaptiveLeaf** | Campo (Enfoque Small-Target) | mAP: 75.0% | N/A | Diseñado específicamente para detectar micro-lesiones en fases muy tempranas que las CNN convencionales ignoran durante el submuestreo de imágenes17. |

La tabla ilustra un paradigma emergente: las arquitecturas **híbridas (CNN-ViT)** representan el estado del arte actual. Mientras que las CNN extraen eficientemente las texturas de los patógenos gracias a sus sesgos inductivos (localidad), los Vision Transformers modelan las relaciones a larga distancia entre múltiples lesiones separadas en la misma hoja mediante mecanismos de autoatención18.

## **Dónde Fallan los Modelos y Limitaciones Documentadas**

A pesar de los avances estadísticos, la bibliografía expone de forma reiterada diversas limitaciones críticas que restringen la confiabilidad de los diagnósticos automatizados en el entorno de producción.

### **1\. El Falso Positivo de la Pre-segmentación (Ruido de Fondo)**

Las investigaciones que comparan imágenes capturadas con cámaras de profundidad (RGB-D) documentan que los modelos entrenados con imágenes sin recortar (pre-segmentación) reportan una precisión artificialmente inflada. El modelo aprende a asociar artefactos del fondo (como el color de un suelo específico o las sombras) con una enfermedad particular7. Al forzar a los modelos a utilizar imágenes "post-segmentación" (donde la hoja se aísla gracias al mapa de profundidad), la exactitud real suele ser marginalmente menor, pero es pragmáticamente más robusta y generalizable a otros cultivos7.

### **2\. Detección de Objetivos Pequeños (Small-Target Detection) y Fases Tempranas**

Un punto de fallo documentado sistémicamente ocurre en los estadios iniciales de la infección. En patologías como el tizón foliar del norte (Northern Leaf Blight) o la mancha gris (Gray Leaf Spot), los síntomas tempranos son minúsculos y carecen de contraste17. Las CNN estándar (como VGG16 o ResNet50) reducen espacialmente la imagen a través de capas de agrupamiento (pooling), lo que provoca la pérdida irremediable de los finos bordes de las lesiones microscópicas. Redes multiescala recientes (como AdaptiveLeaf) han mejorado la retención de estos detalles, pero su precisión media (mAP) sigue estancada alrededor del 75.0% para este tipo de objetivos17.

### **3\. Similitud Inter-clase y Multi-infección**

Los algoritmos experimentan caídas de rendimiento sustanciales al intentar diferenciar patógenos que inducen respuestas fisiológicas similares en el maíz6. Adicionalmente, el supuesto teórico de que una hoja alberga una única enfermedad es falso en la agricultura real. La coexistencia de deficiencias nutricionales, daño por insectos y patógenos fúngicos en una misma muestra foliar confunde a los clasificadores categóricos tradicionales21.

### **4\. Limitaciones de Despliegue (Edge Computing) y la Naturaleza de "Caja Negra"**

Los modelos con mejor rendimiento (como los Ensembles o los ViT de gran escala) requieren potencias de cómputo incompatibles con dispositivos móviles o drones agrícolas23. Existe una fricción tecnológica entre alcanzar precisiones mayores al 99% mediante millones de parámetros (alta demanda de memoria) y desarrollar modelos ligeros (como MobileNetV2 o ConvDeiT-Tiny) que los agricultores puedan ejecutar offline18.  
Asimismo, la falta de explicabilidad (el problema de la "caja negra") disuade la adopción de estas tecnologías. Para que un patólogo confíe en el modelo, requiere mapas de activación (como Grad-CAM o Layer-wise Relevance Propagation) que justifiquen visualmente qué regiones específicas de la hoja fundamentaron la predicción del sistema6.

## **Conclusiones y Consideraciones Previas para Nuevas Investigaciones**

Las evidencias recopiladas demuestran que las investigaciones de vanguardia deben trascender el entrenamiento con conjuntos de datos sintéticos o hiper-controlados. Para que un nuevo estudio aporte valor práctico al estado del arte, debe contemplar los siguientes factores:

1. **Evaluación Cruzada Estricta:** Validar los algoritmos entrenando en bases de datos masivas (como *PlantVillage*) y forzando pruebas de testeo en conjuntos de campo in-situ (como *CD\&S* o *PlantDoc*) para certificar la inmunidad al *domain shift*3.  
2. **Transición a Redes en Cascada o Híbridas:** Abordar las limitaciones visuales separando las tareas funcionales. Primero, segmentar la hoja del fondo (preferiblemente mediante integración multimodal como sensores RGB-D o modelos como LS-RCNN); segundo, clasificar las lesiones combinando la extracción de texturas finas de las CNN con el análisis contextual global de los Vision Transformers10.  
3. **Restricciones de Despliegue:** Optimizar la relación entre el número de parámetros del modelo (GFLOPs) y el F1-Score, priorizando el diseño de arquitecturas ligeras capaces de procesar imágenes complejas en tiempo real directamente en entornos agrícolas aislados17.

## **Referencias Bibliográficas**

Ahmad, A., El Gamal, A., & Saraswat, D. (2023). Toward Generalization of Deep Learning-Based Plant Disease Identification Under Controlled and Field Conditions. *IEEE Access, 11*, 9042-9057.  
Baya, J., et al. (2024). A Mobile-based system for maize leaf disease detection and classification using machine learning algorithms and deep learning: A Case Study of ZeaWatch (Preprint).  
Liu, H., Lv, H., Li, J., Liu, Y., & Deng, L. (2022). Research On Maize Disease Identification Methods In Complex Environments Based On Cascade Networks And Two-Stage Transfer Learning. *Scientific Reports, 12*(1), 18914\.  
Nan, F., Song, Y., Yu, X., Nie, C., Liu, Y., Bai, Y., Zou, D., Wang, C., Yin, D., Yang, W., & Jin, X. (2023). A novel method for maize leaf disease classification using the RGB-D post-segmentation image data. *Frontiers in Plant Science, 14*, 1268015\.  
Shandilya, et al. (2025). Enhanced Maize Leaf Disease Detection and Classification Using an Integrated CNN-ViT Model. *Food Science & Nutrition*.  
Wu, et al. (2026). AdaptiveLeaf: Lightweight Multi-Scale Framework for Small-Target Detection of Maize Leaf Diseases. *16*(13), 1415\.

#### **Obras citadas**

1. Corn leaf disease dataset separated from Plant Village. \- ResearchGate, [https://www.researchgate.net/figure/Corn-leaf-disease-dataset-separated-from-Plant-Village\_tbl2\_377618727](https://www.researchgate.net/figure/Corn-leaf-disease-dataset-separated-from-Plant-Village_tbl2_377618727)  
2. (PDF) Toward Generalization of Deep Learning-Based Plant Disease Identification Under Controlled and Field Conditions \- ResearchGate, [https://www.researchgate.net/publication/367456798\_Towards\_Generalization\_of\_Deep\_Learning-Based\_Plant\_Disease\_Identification\_Under\_Controlled\_and\_Field\_Conditions](https://www.researchgate.net/publication/367456798_Towards_Generalization_of_Deep_Learning-Based_Plant_Disease_Identification_Under_Controlled_and_Field_Conditions)  
3. Toward Generalization of Deep Learning-Based Plant Disease Identification Under Controlled and Field Conditions \- SciSpace, [https://scispace.com/pdf/toward-generalization-of-deep-learning-based-plant-disease-4d3zgln8.pdf](https://scispace.com/pdf/toward-generalization-of-deep-learning-based-plant-disease-4d3zgln8.pdf)  
4. Enhanced Maize Leaf Disease Detection and Classification Using an Integrated CNN‐ViT Model \- PMC, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12208913/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12208913/)  
5. Plant disease recognition datasets in the age of deep learning: challenges and opportunities, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11466843/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11466843/)  
6. MFaster R-CNN for Maize Leaf Diseases Detection Based on Machine Vision | Request PDF, [https://www.researchgate.net/publication/360510907\_MFaster\_R-CNN\_for\_Maize\_Leaf\_Diseases\_Detection\_Based\_on\_Machine\_Vision](https://www.researchgate.net/publication/360510907_MFaster_R-CNN_for_Maize_Leaf_Diseases_Detection_Based_on_Machine_Vision)  
7. A novel method for maize leaf disease classification using the RGB-D post-segmentation image data \- Frontiers, [https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1268015/full](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1268015/full)  
8. Towards Generalization of Deep Learning-Based Plant Disease Identification Under Controlled and Field Conditions \- Purdue e-Pubs, [https://docs.lib.purdue.edu/fund/58/](https://docs.lib.purdue.edu/fund/58/)  
9. Research Advances in Maize Crop Disease Detection Using Machine Learning and Deep Learning Approaches \- MDPI, [https://www.mdpi.com/2073-431X/15/2/99](https://www.mdpi.com/2073-431X/15/2/99)  
10. Research On Maize Disease Identification Methods In Complex Environments Based On Cascade Networks And Two-Stage Transfer Learning \- PubMed, [https://pubmed.ncbi.nlm.nih.gov/36344603/](https://pubmed.ncbi.nlm.nih.gov/36344603/)  
11. Enhanced Maize Leaf Disease Detection and Classification Using an Integrated CNN-ViT Model \- PubMed, [https://pubmed.ncbi.nlm.nih.gov/40599357/](https://pubmed.ncbi.nlm.nih.gov/40599357/)  
12. Enhanced Maize Leaf Disease Detection and Classification Using an Integrated CNN‐ViT Model \- ResearchGate, [https://www.researchgate.net/publication/393185517\_Enhanced\_Maize\_Leaf\_Disease\_Detection\_and\_Classification\_Using\_an\_Integrated\_CNN-ViT\_Model](https://www.researchgate.net/publication/393185517_Enhanced_Maize_Leaf_Disease_Detection_and_Classification_Using_an_Integrated_CNN-ViT_Model)  
13. Enhanced corn seed disease classification: leveraging MobileNetV2 with feature augmentation and transfer learning \- Frontiers, [https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2023.1320177/full](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2023.1320177/full)  
14. (PDF) A Mobile-based system for maize leaf disease detection and classification using machine learning algorithms and deep learning \- ResearchGate, [https://www.researchgate.net/publication/404011009\_A\_Mobile-based\_system\_for\_maize\_leaf\_disease\_detection\_and\_classification\_using\_machine\_learning\_algorithms\_and\_deep\_learning](https://www.researchgate.net/publication/404011009_A_Mobile-based_system_for_maize_leaf_disease_detection_and_classification_using_machine_learning_algorithms_and_deep_learning)  
15. Comparison with Existing Approaches | Download Scientific Diagram \- ResearchGate, [https://www.researchgate.net/figure/Comparison-with-Existing-Approaches\_tbl4\_404011009](https://www.researchgate.net/figure/Comparison-with-Existing-Approaches_tbl4_404011009)  
16. A novel method for maize leaf disease classification using the RGB-D post-segmentation image data \- PubMed, [https://pubmed.ncbi.nlm.nih.gov/37822341/](https://pubmed.ncbi.nlm.nih.gov/37822341/)  
17. AdaptiveLeaf: Lightweight Multi-Scale Framework for Small-Target Detection of Maize Leaf Diseases \- ResearchGate, [https://www.researchgate.net/publication/408213376\_AdaptiveLeaf\_Lightweight\_Multi-Scale\_Framework\_for\_Small-Target\_Detection\_of\_Maize\_Leaf\_Diseases](https://www.researchgate.net/publication/408213376_AdaptiveLeaf_Lightweight_Multi-Scale_Framework_for_Small-Target_Detection_of_Maize_Leaf_Diseases)  
18. ConvDeiT-Tiny: Adding Local Inductive Bias to DeiT-Ti for Enhanced Maize Leaf Disease Classification \- MDPI, [https://www.mdpi.com/2223-7747/15/6/982](https://www.mdpi.com/2223-7747/15/6/982)  
19. A novel method for maize leaf disease classification using the RGB-D post-segmentation image data \- ResearchGate, [https://www.researchgate.net/publication/374244491\_A\_novel\_method\_for\_maize\_leaf\_disease\_classification\_using\_the\_RGB-D\_post-segmentation\_image\_data](https://www.researchgate.net/publication/374244491_A_novel_method_for_maize_leaf_disease_classification_using_the_RGB-D_post-segmentation_image_data)  
20. Crop Disease Image Recognition Based on Transfer Learning \- ResearchGate, [https://www.researchgate.net/publication/322121968\_Crop\_Disease\_Image\_Recognition\_Based\_on\_Transfer\_Learning](https://www.researchgate.net/publication/322121968_Crop_Disease_Image_Recognition_Based_on_Transfer_Learning)  
21. International Journal of, [https://harvardpublications.com/hijiras/article/download/476/477/937](https://harvardpublications.com/hijiras/article/download/476/477/937)  
22. (PDF) CNDD-Net: A Lightweight Attention-Based Convolutional Neural Network for Classifying Corn Nutritional Deficiencies and Leaf Diseases \- ResearchGate, [https://www.researchgate.net/publication/390610625\_CNDD-Net\_A\_Lightweight\_Attention-Based\_Convolutional\_Neural\_Network\_for\_Classifying\_Corn\_Nutritional\_Deficiencies\_and\_Leaf\_Diseases](https://www.researchgate.net/publication/390610625_CNDD-Net_A_Lightweight_Attention-Based_Convolutional_Neural_Network_for_Classifying_Corn_Nutritional_Deficiencies_and_Leaf_Diseases)  
23. (PDF) A novel lightweight hybrid CNN–ViT for maize leaf disease classification, [https://www.researchgate.net/publication/401221123\_A\_novel\_lightweight\_hybrid\_CNN-ViT\_for\_maize\_leaf\_disease\_classification](https://www.researchgate.net/publication/401221123_A_novel_lightweight_hybrid_CNN-ViT_for_maize_leaf_disease_classification)  
24. Identification of maize leaf diseases by using the SKPSNet-50 convolutional neural network model | Request PDF \- ResearchGate, [https://www.researchgate.net/publication/358610306\_Identification\_of\_maize\_leaf\_diseases\_by\_using\_the\_SKPSNet-50\_convolutional\_neural\_network\_model](https://www.researchgate.net/publication/358610306_Identification_of_maize_leaf_diseases_by_using_the_SKPSNet-50_convolutional_neural_network_model)  
25. Corn leaf disease: insightful diagnosis using VGG16 empowered by explainable AI, [https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1402835/full](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1402835/full)  
26. Unveiling Agricultural Insights: Optimizing Transfer Learning Models with Grad-CAM to Improve Maize Disease Detection \- ResearchGate, [https://www.researchgate.net/publication/395125143\_Unveiling\_Agricultural\_Insights\_Optimizing\_Transfer\_Learning\_Models\_with\_Grad-CAM\_to\_Improve\_Maize\_Disease\_Detection](https://www.researchgate.net/publication/395125143_Unveiling_Agricultural_Insights_Optimizing_Transfer_Learning_Models_with_Grad-CAM_to_Improve_Maize_Disease_Detection)