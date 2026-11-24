# Detector de uso de casco de seguridad en obra

**Proyecto Integrador · Desarrollo de Soluciones Inteligentes (02DSI243061) · UBO · 2026 S2**
Equipo: AMIR · Paralelo: D01
Integrantes: Antonia Moya, Isidora Ramos

---

## 1. Problema, usuarios y alcance

- **Problema:** en una obra, verificar que cada trabajador lleve casco depende de que un supervisor lo vea en el momento. Con muchas personas, varios frentes de trabajo y turnos largos, esa revisión es lenta, se cansa y se escapa: quien trabaja sin casco puede pasar horas sin que nadie lo note, y el casco solo protege si se lleva puesto.
- **Usuario:** prevencionista de riesgos o supervisor de obra.
- **Objetivo:** apoyar la inspección avisando cuando una imagen, una cámara o un video muestra a alguien **sin casco**.
- **Alcance:** solo casco / sin casco. No reconoce otros EPP (chaleco, guantes, calzado), no identifica personas y **no reemplaza** la inspección humana.
- **Contexto:** el casco de seguridad protege la cabeza de caídas y de golpes de objetos, pero solo sirve si la persona lo lleva puesto, y en una obra grande nadie puede mirar a todos todo el tiempo. Lo que muestran los estudios:
  - **El casco reduce las lesiones en la cabeza.** Un estudio con 1.298 personas atendidas en urgencias de Corea por caídas en el trabajo encontró lesión dentro del cráneo en el **4,6 % de quienes llevaban casco** (27 de 584) y en el **8,7 % de quienes no** (62 de 714): cerca de la mitad del riesgo. El efecto fue claro en caídas de menos de 4 metros; en caídas más altas no se pudo confirmar, y tampoco se vio diferencia significativa en la mortalidad[^1].
  - **La construcción es donde más se lesiona la cabeza.** La guía de CPWR (2024) indica que la construcción es la industria con más traumatismos cerebrales laborales en EE. UU. Entre 2003 y 2010 murieron 2.210 trabajadores del sector por esta causa, el 25 % de todas sus muertes[^2].
  - **Es un riesgo grande y frecuente.** En EE. UU., entre 2017 y 2021 murieron 25.684 trabajadores de la construcción y 1.079 (4 %) fallecieron por lesiones en la cabeza; más de 6 millones de trabajadores estaban en cargos que exigen protección para la cabeza[^3].
  - **Lo que significa para este proyecto:** como el casco funciona cuando se usa, detectar a tiempo a alguien que no lo lleva es una forma concreta de apoyar la supervisión.

[^1]: Kim, S. C., Ro, Y. S., Shin, S. D. y Kim, J. Y. (2016). *Preventive Effects of Safety Helmets on Traumatic Brain Injury after Work-Related Falls*. Int. J. Environ. Res. Public Health, 13(11), 1063. https://pmc.ncbi.nlm.nih.gov/articles/PMC5129273 (estudio observacional; el uso del casco lo informaron los propios pacientes).
[^2]: CPWR (2024). *Selecting Head Protection for Construction Work*. https://stacks.cdc.gov/view/cdc/213050
[^3]: Weaver, A., Kemp, J., Ojiambo, W. y Simmons, A. (2024). *From Hard Hats to Helmets: The History and Future of Head Protection*. Professional Safety, 69(8), 34-43 (ASSP; datos de BLS).

## 2. Datos y trazabilidad

| Campo | Detalle |
|---|---|
| Dataset | *Hard Hats* v6 (`resized640_noAugmentation-RF-DETR`) |
| Origen | https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5/dataset/6 |
| Workspace | `roboflow-universe-projects` |
| Licencia | CC BY 4.0 (se reconoce la autoría en este README y en el informe) |
| Exportación | 21-01-2026, formato YOLO |
| Tamaño | 19.745 imágenes. Zip original: 13.782 train · 3.962 valid · 2.001 test. La partición usada para entrenar mantiene esos tamaños, pero sin duplicados entre conjuntos (ver abajo) |
| Clases | `Hardhat` (casco) y `NO-Hardhat` (sin casco) |
| Preprocesamiento del autor | Orientación automática (EXIF) y redimensión a 640×640 con *stretch*; sin aumentación |
| Huella del archivo usado | SHA-256 del zip: `d105ed599f42c48b9198dbc93b02a80847c14e2555b4769398778dc92fe7c6f4` |

**Preparación propia:** no se modificaron las imágenes ni las etiquetas del dataset. Se juntaron las 19.745 imágenes del zip y se repartieron de nuevo en *train*, *valid* y *test* **sin duplicados entre conjuntos** (siguiente subsección), con los mismos tamaños que la partición original. El modelo se entrena con *train*, el umbral se elige en *valid* (todas sus imágenes) y la evaluación final se hace en *test*. Durante el entrenamiento, Ultralytics aplica aumentos de datos en memoria (como mosaico, giros y variación de color) que no se guardan en disco.

**Depuración de duplicados (inventario).** Un primer entrenamiento usó la partición original del zip y la evaluación mostró solapamiento entre entrenamiento y prueba, lo que resta independencia a la prueba. Por eso se depuró antes de fijar las particiones (`notebooks/Reentrenar_sin_duplicados.ipynb`):

- **Método:** huella perceptual de 16×16 píxeles en escala de grises (256 bits, cada píxel comparado con la media de la imagen). Dos imágenes se consideran casi idénticas si su huella difiere en 12 bits o menos (0 = idénticas). Se agrupan las imágenes conectadas por esa relación y se reparten **grupos completos** (semilla 42), de modo que una misma escena no queda en dos conjuntos.
- **Resultado:** 99.488 pares casi idénticos; 15.384 grupos, de los cuales 1.538 tienen más de una imagen y reúnen 5.899 imágenes (29,9 % del dataset).
- **Test original:** 410 de sus 2.001 imágenes (20,5 %) pertenecían a un grupo con alguna imagen de *train*.
- **Después de repartir:** 0 pares casi idénticos entre conjuntos distintos (el notebook lo verifica y se detiene si no se cumple).

| Partición original → nueva | test | train | valid |
|---|---|---|---|
| test | 217 | 1.375 | 409 |
| train | 1.377 | 9.699 | 2.706 |
| valid | 407 | 2.708 | 847 |

| Cajas por clase (partición nueva) | train | valid | test |
|---|---|---|---|
| Hardhat | 27.957 | 9.482 | 4.989 |
| NO-Hardhat | 9.599 | 2.120 | 1.246 |

Inventario completo (una fila por imagen: partición original, partición nueva y grupo): `evidencia/inventario_duplicados_y_particion.csv`. **Límites del método:** la huella es una aproximación; puede no detectar escenas del mismo video con otro encuadre y puede agrupar imágenes distintas pero parecidas. El reparto fue por grupos y no por clase, así que la proporción de cajas "sin casco" varía entre conjuntos (25,6 % en *train*, 18,3 % en *valid* y 20,0 % en *test*).

**Limitaciones de los datos:** las imágenes están estiradas a 640×640, mientras que un video real tiene otra proporción, así que el rendimiento en cámara real puede ser distinto. El dataset es de terceros y puede no representar las obras del contexto del equipo. Solo contiene dos clases. Varias imágenes no son de obra (reunión en oficina, grupos con mascarillas, fotos de stock), hay personas con el rostro identificable y faltan algunas etiquetas (personas sin casco sin caja). Además, el dataset original traía solapamiento entre conjuntos (ver la depuración de arriba), que se corrigió en la partición usada.

## 3. Técnica / modelo y decisión

- **Modelo usado:** **YOLO26n** (Ultralytics, tamaño *nano*), ajustado por el equipo con el dataset Hard Hats v6 a partir de los pesos preentrenados `yolo26n.pt` (*transfer learning*), con la partición sin duplicados. Configuración: hasta 30 épocas (se completaron las 30, en 2,075 h), imágenes de 640 px, batch 16, paciencia 10, semilla 42, optimizador elegido automáticamente por Ultralytics (AdamW, tasa inicial 0,001667), GPU T4 de Colab. El resultado es `modelo_casco_yolo26.pt` (5,5 MB). Se ejecuta **sin API key y sin cuenta de Roboflow**, y para analizar no usa ningún servicio externo de inferencia. Código: `notebooks/Reentrenar_sin_duplicados.ipynb`. El primer entrenamiento, con la partición original (`notebooks/Entrenamiento_YOLO26.ipynb`), se conserva como antecedente: completó 23 épocas en 1,388 h y dio mAP50 0,947 en un test con solapamiento.
- **Por qué este modelo:** (1) detecta exactamente las dos clases del problema; (2) no necesita API key ni créditos para inferir, lo que hace la demo reproducible por cualquier persona con el archivo `.pt`; (3) es rápido (en la evaluación de Ultralytics con GPU T4: 1,1 ms de preprocesado, 3,5 ms de inferencia y 1,5 ms de postprocesado por imagen, en lotes) y liviano (2,4 millones de parámetros); (4) el dataset es público y con licencia abierta (CC BY 4.0).
- **Alternativa considerada (comparación hecha con el modelo inicial y la partición original, en 200 imágenes de *test*; no se repitió con el modelo actual):**

| Modelo | Umbral | P sin casco | R sin casco | F1 sin casco | Alerta por imagen (de 40 con alguien sin casco) | Dependencias |
|---|---|---|---|---|---|---|
| YOLO26n propio, modelo inicial | 0,40 | 0,936 | 0,914 | 0,925 | recall 0,975 (1 perdida) · precisión 0,975 (1 falsa alarma) | ninguna externa |
| Modelo alojado en Roboflow (`hard-hats-fhbh5/6`) | 0,30 | 0,906 | 0,977 | 0,940 | recall 1,00 (0 perdidas) · 2 falsas alarmas | internet, API key y créditos |

  El modelo alojado rinde algo mejor en recall de "sin casco" (0,977 frente a 0,914), pero se desconoce con qué imágenes fue entrenado, así que su resultado puede estar inflado; además exige API key, internet y créditos, y es más lento por ser una consulta remota. Con una muestra de solo 128 cajas de "sin casco" la diferencia puede caber en el margen de error. Se eligió el modelo propio por reproducibilidad y autonomía, aceptando este costo. Protocolo: sección 4 de `notebooks/Entrenamiento_YOLO26.ipynb`, que compara ambos modelos en las mismas imágenes con las mismas reglas (IoU ≥ 0,5) y lee las predicciones del modelo alojado desde `pred_test.json` (generado por la bitácora; guardado en `evidencia/`).
- **Decisión de umbral:** se elige en *valid* y se evalúa en *test* una sola vez. Se tomó el máximo de la curva F1 de "sin casco" (NO-Hardhat, suavizada con una media móvil de 25 puntos) sobre las 3.962 imágenes de *valid* (11.602 cajas). **Umbral elegido: 0,37.** Es la configuración inicial de la demo y se puede ajustar con un control deslizante (pasos de 0,01); con cámara real puede convenir bajarlo (ver sección 7).
- **Filtro de solapamiento:** como medida de precaución, la demo conserva solo la caja de mayor confianza cuando dos cajas de clases distintas se solapan (IoU ≥ 0,5). Con el modelo propio no se observó que marque la misma cabeza con las dos clases.
- **Lógica de alerta:** un cuadro con "sin casco" es *POSIBLE*. Se confirma como *ALERTA* si la **misma persona** sale sin casco en **2 cuadros analizados consecutivos**. "Misma persona" significa que la caja "sin casco" de un cuadro se solapa (IoU ≥ 0,3) con una caja "sin casco" del cuadro anterior; no se reconoce a nadie, solo se sigue la posición de la caja. La alerta termina tras 3 cuadros seguidos sin "sin casco", y cada persona confirmada queda como un evento. Esta regla reemplaza la anterior ("2 de los últimos 3 cuadros"), que podía confirmar con detecciones no consecutivas y sin verificar que fueran la misma persona. Costo: con un cuadro cada 2 s, una persona que se mueve mucho puede no solaparse entre cuadros y la alerta tardaría o se perdería.

## 4. Arquitectura e integración

![Diagrama de arquitectura: entrada, redimensión, modelo YOLO26n, filtro de umbral y solapamiento, persistencia por persona (2 cuadros consecutivos) e interfaz Gradio](arquitectura.png)

La app (interfaz y modelo) corre en un mismo lugar, que es un Space de Hugging Face; no usa servicios externos de inferencia.

Interfaces y datos: la imagen entra como archivo, cuadro de cámara o video; el modelo devuelve cajas con clase y confianza; la app entrega estado, imagen anotada, tabla de eventos (hora o minuto del video) y CSV. En el Space público no se guardan capturas ni imágenes.

| Componente | Archivo | Función |
|---|---|---|
| Entrenamiento con partición sin duplicados (principal) | `notebooks/Reentrenar_sin_duplicados.ipynb` | Calcula huellas, agrupa duplicados, reparte por grupos, entrena YOLO26n, evalúa en *test*, elige el umbral en *valid* y guarda el `.pt`. |
| Entrenamiento inicial (antecedente) | `notebooks/Entrenamiento_YOLO26.ipynb` | Primer entrenamiento con la partición original y comparación con el modelo alojado. |
| Modelo | `hugging face/modelo_casco_yolo26.pt` | Pesos usados por la demo (los del entrenamiento sin duplicados). |
| Demo publicada (principal) | Carpeta `hugging face/` (`app.py`, `modelo_casco_yolo26.pt`, `requirements.txt`, `packages.txt`, `README.md`) | La app disponible con un enlace fijo, sin instalar nada: https://huggingface.co/spaces/AntoM07a/Detector-Casco |
| Validación del hito 30-10 | `notebooks/Bitacora_Validacion.ipynb` | Validación con el modelo alojado en Roboflow (requiere API key): contrato, 12 casos de prueba, v1 frente a v2, robustez y bitácora. |
| Evidencia | `evidencia/` | Inventario de duplicados y métricas del *test* sin duplicados. |

## 5. Requisitos y cómo ejecutar

**Requisitos:** para usar la demo publicada, solo un navegador (con cámara para el modo en vivo). **No necesita API key.** Reproducir el entrenamiento requiere una cuenta de Google (Colab) y una GPU (la T4 gratuita sirve).

**Dependencias:** `ultralytics`, `gradio` (≥ 5), `opencv-python`, `pandas`, `pillow` y `numpy` (`hugging face/requirements.txt`). Colab ya trae todas menos `ultralytics`; **no reinstalar `pillow`** (rompe el entorno). Si aparece `ImportError`: *Entorno de ejecución → Reiniciar sesión* y ejecutar desde la celda de configuración. El video anotado usa `ffmpeg` si está instalado.

**Pasos: demo publicada (Hugging Face Spaces)**
1. Abrir https://huggingface.co/spaces/AntoM07a/Detector-Casco en el navegador. Si lleva varios días sin uso, el Space está dormido y tarda cerca de un minuto en despertar.
2. En la pestaña "Cámara en vivo", permitir el acceso a la cámara; o en "Video grabado", subir un video y presionar "Analizar video".
3. Para volver a publicarla: crear un Space de tipo Gradio en Hugging Face (hardware ZeroGPU, gratuito), subir los archivos de la carpeta `hugging face/` y dejarlo en visibilidad pública. La app corre en CPU; la función `@spaces.GPU` que contiene solo cumple el requisito de la plataforma y no se usa.

**Pasos: reproducir el entrenamiento**
1. Dejar el dataset en Drive, como zip o como carpeta descomprimida, y poner su ruta en `DATASET_ZIP` de `notebooks/Reentrenar_sin_duplicados.ipynb`.
2. Activar GPU (T4) y ejecutar todas las celdas en orden. El entrenamiento tardó cerca de 2,1 h; los pesos, el inventario y las métricas quedan en la carpeta de salida de Drive (`SALIDA`).

**Opcional (modelo alojado):** el notebook `notebooks/Bitacora_Validacion.ipynb` requiere `inference-sdk`, una cuenta gratuita de Roboflow y su API key guardada en *Secretos* de Colab como `ROBOFLOW_API_KEY`. Nunca escribir la clave en una celda.

## 6. Pruebas y resultados

Cifras de la ejecución del equipo de `notebooks/Reentrenar_sin_duplicados.ipynb` (GPU T4), con el modelo publicado y la partición sin duplicados.

- **Test completo (2.001 imágenes, 6.235 cajas, mAP de Ultralytics).** P y R son los del punto de mejor F1 de ese conjunto; mAP50 y mAP50-95 no dependen de ese punto.

| Clase | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Todas | 0,915 | 0,903 | 0,945 | 0,602 |
| Hardhat | 0,913 | 0,919 | 0,949 | 0,579 |
| NO-Hardhat | 0,917 | 0,887 | 0,940 | 0,626 |

- **Precisión y recall en *test* con el umbral elegido en *valid* (0,37, IoU 0,5):** Hardhat P 0,920 · R 0,914; NO-Hardhat (sin casco) P 0,922 · R 0,879. Este es el punto de operación de la demo; el umbral no se cambió después de ver *test*.
- **Validación (3.962 imágenes, 11.602 cajas):** P 0,917 · R 0,885 · mAP50 0,938 · mAP50-95 0,604.
- **Comparación con el modelo inicial.** Con la partición original el test daba mAP50 0,947 y mAP50-95 0,602 (P 0,921, R 0,899). Con una prueba sin duplicados da 0,945 y 0,602, es decir, casi no cambió. Esto sugiere que el solapamiento no estaba inflando mucho la métrica, pero **no es un experimento controlado**: el test nuevo contiene solo 217 imágenes del test original, y el entrenamiento nuevo corrió 30 épocas frente a 23 del inicial.
- **Velocidad:** en la evaluación de Ultralytics con GPU T4 (lotes), 1,1 ms de preprocesado, 3,5 ms de inferencia y 1,5 ms de postprocesado por imagen. En el Space público la app corre en CPU compartida y no se midió su velocidad.
- **Comparación con el modelo alojado en Roboflow:** solo se hizo con el modelo inicial (ver sección 3); no se repitió con el modelo actual.
- **Pruebas de integración y robustez con el modelo alojado (hito 30-10, `notebooks/Bitacora_Validacion.ipynb`):** 12 casos de prueba definidos antes de ejecutar; la app v1 cumple 1 de 12 y la v2 cumple 11 de 12 (CP-11, comparación exacta con la consulta directa, falló en 1 de 5 imágenes sin causa identificada). Casos límite probados: archivo que no es imagen, imagen dañada, campo ausente, umbral 0 y 1,5, imagen de 16 px (rechazada) e imagen gris sin personas. Robustez en 30 imágenes con "sin casco": con baja resolución (50 %) se mantuvieron las 30 alertas; con poca luz (brillo 40 %) se perdió 1. **Estas pruebas se hicieron con el modelo alojado, no con el modelo propio, y no se repitieron con este.**
- **Observación con el modelo inicial (cámara en vivo, persona con un gorro):** con umbral 0,9 no se detectó nada; con 0,5 se detectó como **POSIBLE** sin casco; con 0,3 se detectó **sin casco**. Es una observación cualitativa, no una métrica, y no se repitió con el modelo actual. Sugiere que un umbral muy alto puede dejar de reportar a la persona.
- **Falsas alertas y omisiones en video anotado: no se midieron.** Con el modelo actual no se anotó ningún video cuadro a cuadro, así que no hay cifra de falsas alertas ni de omisiones de la app en video. Un cuadro con alguien sin casco que aún no completa 2 cuadros consecutivos contaría como omisión; es un límite de la regla.
- **Evidencia:** notebooks ejecutados con salidas visibles, `evidencia/inventario_duplicados_y_particion.csv`, `evidencia/metricas_test_sin_duplicados.json`, `eventos_camara.csv` y `eventos_video.csv` (generados por la app), y la captura de la app publicada funcionando.

## 7. Limitaciones conocidas y riesgos

- **No es tiempo real continuo:** la demo analiza un cuadro cada 2 s (ajustable); pueden perderse eventos más cortos, y la alerta exige 2 cuadros consecutivos, así que se confirma al menos 2 s después de que aparece la persona. La velocidad depende de si hay GPU; en el Space público corre en CPU compartida y puede ir más lenta, y si lleva días sin uso tarda en despertar.
- **Falsos negativos:** una persona sin casco no alertada es el error más grave. En *test*, con el umbral de la demo, el recall por caja de "sin casco" fue 0,879 (cerca de 12 de cada 100 cajas sin casco no se detectan). Pueden ocurrir con poca luz, distancia, oclusión, ángulos raros o cascos poco comunes.
- **Falsas alarmas:** gorras, capuchas u otros objetos pueden confundirse. La persistencia temporal las reduce, pero no las elimina.
- **Límites de la evaluación:** las métricas se calculan en una partición sin duplicados, pero las imágenes de los tres conjuntos vienen de la misma fuente y la depuración es aproximada (huella perceptual con umbral 12), así que pueden quedar escenas parecidas entre conjuntos. El rendimiento en una cámara real distinta (otra luz, otra proporción de imagen) puede ser peor; por eso la demo permite bajar el umbral. La proporción de cajas "sin casco" varía entre conjuntos (ver sección 2).
- **Seguimiento por posición:** la regla "misma persona" compara cajas entre cuadros con IoU ≥ 0,3 y supone cámara fija. Con 2 s entre cuadros, una persona que se mueve mucho puede perder la coincidencia, y dos personas que se cruzan pueden confundirse.
- **Comparación con el modelo alojado:** se hizo con una muestra de 200 imágenes (128 cajas de "sin casco") y con el modelo inicial; la diferencia puede caber en el margen de error.
- **Alcance de las pruebas:** los casos de prueba y la robustez se ejecutaron con el modelo alojado, no con el propio (ver sección 6). La robustez cubrió solo dos cambios (poca luz y baja resolución). Las falsas alertas y omisiones en video anotado no se midieron (sección 6).
- **Privacidad:** en el Space público, las imágenes de quien lo usa se envían al servidor de Hugging Face donde corre la app; la app no guarda capturas ni imágenes. El Space es público: cualquier persona con el enlace puede usarlo, por lo que no debe compartirse fuera del curso, y debe haber autorización de las personas filmadas, sin buscar identificarlas.
- **Seguridad:** la demo no usa claves. No incluir claves, capturas con personas identificables ni datos personales en notebooks o repositorios. Como el Space es público, su código y el modelo son visibles para cualquiera.
- **Uso:** el resultado es orientativo y no reemplaza la inspección de seguridad.

## 8. Licencias y créditos

- Dataset: *Hard Hats* v6, Roboflow Universe Projects, CC BY 4.0 — https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5/dataset/6
- Modelo: YOLO26n de Ultralytics (pesos preentrenados `yolo26n.pt`), ajustado por el equipo. Ultralytics se distribuye bajo licencia AGPL-3.0 (y licencia empresarial); el uso en este trabajo es académico. El Space público publica el código y el modelo, lo que es coherente con esa licencia.
- Estructura de pruebas y validación: Clase 11 de Desarrollo de Soluciones Inteligentes (UBO), adaptada al detector de cascos.
- Plataforma de publicación: Hugging Face Spaces.
- Librerías (licencias según los metadatos de cada paquete en PyPI, consultados el 06-11-2026): `ultralytics` (AGPL-3.0), `gradio` (Apache-2.0), `opencv-python` (Apache-2.0), `pandas` (BSD-3-Clause), `numpy` (BSD-3-Clause, con componentes bajo otras licencias permisivas), `pillow` (MIT-CMU, equivalente a HPND), `torch` (BSD-3-Clause, con componentes bajo otras licencias permisivas) y `spaces` (Apache-2.0, solo en el Space). Solo para el modelo alojado: `inference-sdk`, `reportlab`, `openpyxl` (licencias no verificadas).

## 9. Declaración ampliada de uso de IA generativa

| Aspecto | Detalle |
|---|---|
| Herramienta | Claude (Anthropic) |
| Finalidad | Apoyo en código y estructura de los notebooks y de la app, diagrama, redacción de este README, borrador de textos de análisis de la bitácora, propuesta de métricas y del protocolo de comparación de modelos, preparación de los archivos para publicar la app en Hugging Face Spaces, y, para la corrección posterior al EVAL-01: el notebook de depuración de duplicados y reentrenamiento, y el seguimiento de la misma persona en `app.py`. |
| Componentes afectados | `Entrenamiento_YOLO26.ipynb`, `Reentrenar_sin_duplicados.ipynb`, `app.py` (versión del Space), `Bitacora_Validacion.ipynb`, el diagrama de arquitectura y este README. |
| Verificación realizada | El equipo ejecutó los notebooks en Colab con el dataset y el modelo reales; las cifras de este README provienen de esas ejecuciones (incluida la salida de `Reentrenar_sin_duplicados.ipynb`). |
| Cambios humanos | El equipo decidió usar un modelo propio en lugar del modelo alojado en Roboflow, para no depender de API key ni de internet. Se entrenó el modelo en Colab y se ejecutaron todos los notebooks con el dataset real; además se configuraron las rutas de Drive (AMIR/DATASET_ZIP) y el umbral, y se publicó la app en un Space de Hugging Face con la cuenta del equipo. Ejecutamos el reentrenamiento sin duplicados en Colab y subimos sus resultados al repositorio. Escribimos los textos de análisis de la bitácora. Editamos el README, elegimos qué contexto y qué fuentes dejar en el README (solo estudios sobre cascos de seguridad) y revisamos que las cifras coincidieran con las ejecuciones. |

Nota: Claude probó la lógica de la demo y de la partición por grupos con datos sintéticos, sin acceso al dataset real, a GPU ni a Hugging Face. Los resultados reales provienen únicamente de las ejecuciones del equipo.

## 10. Estructura del repositorio

```
├── README.md
├── arquitectura.png                      # diagrama de la sección 4
├── hugging face/                         # archivos de la demo publicada en Hugging Face Spaces
│   ├── app.py
│   ├── modelo_casco_yolo26.pt            # pesos del entrenamiento sin duplicados
│   ├── requirements.txt
│   ├── packages.txt
│   └── README.md
├── notebooks/
│   ├── Reentrenar_sin_duplicados.ipynb   # depuración, entrenamiento, evaluación y umbral (principal)
│   ├── Entrenamiento_YOLO26.ipynb        # primer entrenamiento, partición original (antecedente)
│   └── Bitacora_Validacion.ipynb         # validación del hito 30-10 (modelo alojado)
└── evidencia/
    ├── inventario_duplicados_y_particion.csv
    ├── metricas_test_sin_duplicados.json
    └── pred_test.json                    # predicciones del modelo alojado usadas en la comparación
```

Antes de entregar: confirmar que los archivos finales coinciden con la versión que se demostrará, que no hay claves ni datos personales, y que todos los **COMPLETAR** fueron reemplazados.
