### Tema: Desarrollo de una red neuronal recurrente para la generación de datos para diarios.

 #### Autores:

Abel Antonio Cruz Suárez C-411

Yan Carlos González Blanco C-411

José Carlos Hernández Piñera C-411

Henry Estevez Gómez C-411



**Requisitos**

- instalar la biblioteca:

  - $keras$

  - $tensorflow$
  - $foreign$
  - $TraMineR$
  - $python \space 3.x$

#### Aplicación 

El problema en cuestión es la generación de datos nuevos a partir de un conjunto de actividades diarias de varios individuos. Para darle solución a este problema se implementó un modelo de aprendizaje de máquinas, de forma tal que sea alimentado con un set de datos sobre el tema que se está indagando y luego sea capaz de generar secuencias nuevas parecidas a aquellas con las que fue entrenada esta red neuronal. 

#### Modo de uso

Al usuario se le provee un archivo denominado $pretrainedModel.R$ que constituye un modelo previamente entrenado y listo para la generación de datos nuevos. Solo es necesario cargar el set datos con que se comienza a trabajar y al correr el archivo obtendrá la nueva secuencia. Cargar el set de datos es necesario pues al procesarlo se crea la fuente de información de donde se crea la nueva secuencias. El resultado de correr este script es una salida en consola y dos archivos, un $csv$ y un $txt$ con las actividades generadas.

En este script se encuentra el algoritmo encargado de la generación, si se desea cambiar la palabra inicial con la que se comienza la creación de datos solo hay que cambiar la actividad en $seed\_text$. Se recomienda mantener la que está actualmente ($Sleeping$), pues constituye un buen inicio para del día de una persona. 

```R
start_i <- 167
stop_i <- 310
size <- 30
seed_text <- "Sleeping"
```

La anterior sección de código muestra los datos que pueden ser cambiados en para la generación. Recordar que $start\_i$ y $stop\_i$ deben coincidir con los del proceso de la configuración de la red neuronal y  $seed\_text$ debe ser una actividad real de las que hay en $map\_seqs2char$.

**Nota**

En la carpeta checkpoints se encuentra los coeficientes del modelo entrenado. Los archivos $generator\_model.h5$ y $training\_model.h5$ son los modelos de generación y de entrenamiento. Estos archivos no deberían borrarse, pues son cargados por la aplicación. También se provee la base de datos con que se trabajó.

**Configuración de la red neuronal**

Este modelo fue entrenado con una base de datos específica y con sus secuencias particulares, por tanto, no generará datos que no sean según la distribución aprendida. A pesar de que aquí se entrega un generador de secuencias totalmente funcional, si se desea, se puede acceder al archivo $main.r$ y realizar desde cambios en los hiperparámetros, procediendo a ejecutar otros entrenamientos; hasta cambios en los datos, proveyendo una nueva base de datos de donde extraer la información. Si realiza esta última acción debe conocer exactamente cuáles son las columnas donde se encuentran las actividades, para poder extraerlas de dicha base de datos y realizar los trabajos pertinentes de aprendizaje. En la base de datos actual se trabaja desde la columna $167$ hasta la $310$, que constituyen las actividades primarias. Como se aprecia en el código siguiente extraído del script. 

```R
start_i <- 167
stop_i <- 310
```

Una vez que se haya completado el proceso de extracción y modificación de los datos podemos cambiar el tamaño de las secuencias con las que se quiere que aprenda el modelo y la cantidad de las mismas. Como se muestra en el siguiente código, se están utilizando $4000$ secuencias compuestas cada una por $100$ actividades. Para este caso no se recomienda usar una cantidad distinta de las $4000$ secuencias pues pudieran no coincidir las dimensiones del vector de entrada con la arquitectura planteada. Lo que se pudiera hacer si se desean utilizar todas las secuencias ($4360$) es emplear solo $batches$ de tamaño $8$, pero en este se decidió no utilizar dichos parámetros para poder trabajar y analizar otros tamaños de $batches$.

En caso de querer trabajar con los hiperparámetros puede cambiar los siguientes valores:

```R
batch <- 32
emb_dim <- 64 
rnn_u <- 128
learning_rate<-0.0001
epochs <- 55
validation_perct<-0.2
```

Después de varios experimentos para encontrar el modelo más preciso, los anteriores hiperparámetros se consideraron los más eficientes. En esta decisión influyen muchas cuestione; por ejemplo seleccionar una mayor cantidad de $recurrent \space neural \space network \space units$ ($rnn\_u$) contribuirá a un mayor conocimiento de los patrones subyacentes en los datos, pero también tributa a que exista un mayor $overfitting$ en el modelo y el poder de cómputo es fundamental para aumentar dicha cantidad, pues de lo contrario se tarda mucho el proceso de aprendizaje si se incrementa este valor. Otro parámetro importante para que la red "aprenda" más sería incrementar el número de $epochs$. 

La conclusión final es que reajustando dichos parámetros para obtener mejores resultados nunca se llegó a sobrepasar el $90\%$ de precisión, para ello se llegó a trabajar con una mayor cantidad de $rnn\_u$, $emb\_dim$ y de $epochs$. Mientras que con el modelo actual se alcanza un $88\%$ de precisión aproximadamente.

Después de haber reajustado los parámetros y los datos se procede a compilar la arquitectura planteada con los nuevos valores y a ejecutar el proceso de entrenamiento. Automáticamente, se guarda dicho modelo. Los valores del modelo guardado deben ser cargados en una nueva red que solo cambia la forma en que recibe las entradas y es la utilizada para generar los nuevos datos. Este nuevo modelo se guarda y luego corriendo el archivo $pretrainedModel.R$ se obtienen los resultados.

Para este trabajo se emplea una red neuronal recurrente basada en $LSTM$ y que construimos a través de un llamado al método $build\_model\_lstm$. Si se desea se puede usar una red neuronal simple sin el algoritmo $LSTM$ que también se construyó mediante un llamado a $build\_model\_simple\_rnn$.

#### Arquitectura de la red neuronal

Para construir este modelo de aprendizaje automático se usa el algoritmo $Long Short-Term
Memory \space (LSTM)$, que es fundamental para el trabajo con series de tiempo. Debemos destacar que se deja la arquitectura de una red neuronal recurrente sin este algoritmo  ($LSTM$) que resultó ser igual de eficiente para este problema, solo que por una cuestión de que se pudiera trabajar con datos distintos o con mayor dependencia entre ellos, se mantiene el uso de $LSTM$. La razón por la que estas dos arquitecturas tienen resultados similares (en dependencia de los hiperparámetros), se debe a la distribución de los datos y que de por sí no son muchos. Por ejemplo, en muchos casos obtenemos secciones de secuencias de la siguiente forma $Sleeping$, $Sleeping$, $Sleeping$, $washing \space and \space dreasing$, $washing \space and \space dreasing$, ... y secuencias de ese estilo no resultan muy difíciles de procesar, ni entender su relación son otras secciones pasadas.

Se debe aclarar que al usar las redes neuronales recurrentes, en cada una de sus unidades reciben información de estados anteriores y de su estado actual. La utilización de $LSTM$ permite combatir el problema del $vanishing-gradient$ o gradiente que se desvanece, que hace que a medida que agreguemos más capaz $RNN$ se va haciendo imposible de entrenar, principalmente debido a que los pesos que se van ajustando en la red se hacen muy cercanos a cero. Esencialmente, lo que hace $LSTM: $ es guardar información para más adelante, evitando que las señales más antiguas desaparezcan gradualmente durante el procesamiento. Para esto son agregadas las unidades de la red nuevas operaciones sobre matrices que permiten olvidar deliberadamente información irrelevante en el flujo de datos y otras son las encargadas de actualizar la información sobre el presente, actualizando la el flujo de información pasada con nueva información.

##### Diseño

![lstm arch](/media/abelo/Local Disk/4to/Mio/2do Semestre/MO II/project/lstm arch.png)

Como se aprecia en la imagen anterior se recibe una secuencia de frases (actividades), anteriormente vectorizadas, y son pasadas a una capa embebida. En esta primera capa es donde se puede asociar una palabra a un vector $(word \space embeddings)$.  Estos vectores son de punto flotante de baja dimensión y sus valores son aprendidos de los datos. Mientras mayor sea el vocabulario se tiende a usar vectores de una mayor dimensión $256$, $512$ o $1024$. 

Una capa embebida se puede ver como un diccionario que mapea índices enteros (que representan palabras específicas) a vectores densos. Toma números enteros como entrada, busca estos enteros en un diccionario interno, y devuelve los vectores asociados. Al ser entrenada como una capa más de la red, se trata de conocer la estructura y distribución de dichos vectores, de forma tal que si hay palabras o frases con significados parecidos, no estarán muy dispersos sus vectores en el espacio geométrico.

Estos vectores son los que alimentan a la capa $LSTM$ explicada con anterioridad y los resultados de esta son enviados a una capa densa encargada de dar la clasificación. Esta clasificación no es más que, para una frase determinada cual es su más probable próxima frase.  Se deben ver a las frases como posibles categorías y dada una entrada (frase) se desea determinar lo más preciso posible a que categoría pertenece. Donde decir que pertenece a una categoría es determinar cuál es su próxima frase. Mediante el entrenamiento de la red utilizando el algoritmo de optimización del gradiente descendiente se busca el conjunto de coeficientes que dan como resultado la mejor estimación de la función objetivo. Más específicamente se usa el algoritmo gradiente descendiente estocástico. En esta variación, se ejecuta el procedimiento de descenso de gradiente, pero la actualización de los coeficientes se realiza para cada instancia de entrenamiento, en lugar de al final del lote de instancias. 

Como la función sobre la que nos encontramos trabajando es diferenciable, efectuamos el cálculo de los valores del gradiente, mediante la regla de la cadena. Este algoritmo es conocido como $backpropagation$, el mismo comienza con el valor de pérdida final y funciona hacia atrás desde las capas superiores a las capas inferiores, aplicando la regla de la cadena para calcular la contribución que cada parámetro tenía en el valor de la pérdida. Mientras menor sea la pérdida más preciso será nuestro modelo.

Para mitigar los problemas de overfitting empleamos una red que no fuera demasiado grande. Es por ello que solo se hace uso de una capa embebida con vectores de solo $64$ dimensiones, una $LSTM$ con $64$ unidades y otra densa para obtener los resultados de $39$ unidades (representando cada una de las $38$ frases, desde la $0$). De esta forma si la red tiene recursos de memorización limitados, no será
capaz de aprender el mapeo tan fácilmente; por lo tanto, para minimizar su pérdida, tendrá que
recurrir al aprendizaje de representaciones comprimidas que tienen poder predictivo con respecto a los objetivos. Al mismo tiempo no estamos usando tan pocas unidades de forma que carezca de recursos para la memorización.

Otra técnica empleada fue la regularización de pesos, mediante la cual se imponen restricciones a la complejidad de una red al obligar a sus pesos a tomar solo valores pequeños, lo que hace que la distribución de valores de peso sea más regular. Se empleó $Regularización \space L2$, donde el costo agregado es proporcional al cuadrado del valor de los pesos de los coeficientes. En el script de la arquitectura podemos apreciar lo siguiente: 

```R
kernel_regularizer = regularizer_l2(0.001)
```

que significa que cada coeficiente en la matriz de peso de la capa agregará $0.001 * weight\_coefficient\_value$ a la pérdida total de la red.

Finalmente, la última técnica aplicada fue $Dropout$ o abandono. Esta técnica de regularización, aplicado a una capa, consiste en aleatoriamente establecer en cero una serie de valores de salida de la capa durante el entrenamiento. Para ello se escoge entre un $20\%$ a $50\%$. De esta forma se fuerza a la red a no confiarse de ningún nodo y se evita que se aprenda patrones que son significantes y que de no agregar este ruido lo haría.

Para la otra arquitectura se utilizan las mismas técnicas solo que se tienen dos capas de redes neuronales recurrentes simples en lugar de una $LSTM$.

En la siguiente imagen se puede apreciar la arquitectura fabricada, junto con las dimensiones de cada capa y los parámetros a entrenar.

![Screenshot from 2022-02-13 01-17-36](/media/abelo/Local Disk/4to/Mio/2do Semestre/MO II/project/Screenshot from 2022-02-13 01-17-36.png)



En esta imagen se muestra la arquitectura del modelo encargado de generar los datos. A diferencia del anterior este recibe únicamente una actividad y genera la próxima según la distribución aprendida y esta nueva genera la próxima, hasta obtener todas las deseadas.

![Screenshot from 2022-02-13 01-46-42](/media/abelo/Local Disk/4to/Mio/2do Semestre/MO II/project/Screenshot from 2022-02-13 01-46-42.png)

#### Flujo de trabajo

En la siguiente imagen se hace un resumen de como se lleva a cabo el flujo de trabajo, desde que se obtienen los datos hasta que se generan las nuevas distribuciones.

![Screenshot from 2022-02-13 00-57-50](/media/abelo/Local Disk/4to/Mio/2do Semestre/MO II/project/Screenshot from 2022-02-13 00-57-50.png)



#### Módulo Utils

En este módulo se encuentran aquellas funciones básicas empleadas para el procesamiento de los datos. Con ellas se logra partir de un dataset cualquiera, acceder a la información que se desea y convertirla en un set de entrenamiento.

$fix\_seq$ es una función muy particular que se crea solo para que no cuando se extraigan las actividades, se obtenga solo un string con los caracteres que la conforman, eliminando los espacios en blanco al final de la secuencia extraída que estén de sobra.

$convert\_dataSet$ es la función encargada de extraer las actividades del dataset entregado a partir de la columna inicio que escojamos hasta la final.

$clean\_data$ es la encargada de eliminar los espacio demás de un conjunto de secuencias.

$map\_seq2index$ es con la función que obtenemos las actividades, únicas del set de actividades, las procesamos y  generamos un diccionario de actividades. De tal forma que si se desea saber el número que representa una actividad, se indexa dicha actividad y se obtiene. De forma contraria podemos obtener las actividades con el uso de la función $names$ de R.

$make\_tensor$ permite, usando el diccionario de actividades y el data set de las mismas, obtener un nuevo data set con los valores que representan.

$get\_batch$ Es la función que permite construir el set de entrenamiento. Para ello se seleccionan de modo aleatorio $batch\_size$ secuencias de actividades del set vectorizado y luego de forma aleatoria se toma un índice de dicha secuencia para producir uno de los $batch\_size$ datos de entrada con que se entrena la red. Los datos con que se evalúa la red neuronal son los mismos, solo que corridos una posición a la derecha. Ejemplo, si tenemos $input=[x_1,x_2,x_3,x_4]$ entonces se tendrá $output=[x_2,x_3,x_4,x_5]$. Y así se repite el proceso hasta obtener una matriz de tamaño $batch\_size\space X \space seq\_length$.

$text\_generation$ esta función es la empleada para guardar y generar los datos.

#### Particularidades

La falta de una mayor cantidad de datos y de un equipo de cómputo más potente no permiten obtener mejores resultados que los que se ofrecen. A pesar de estos inconvenientes se logra un generador de datos de actividades con una precisión durante el entrenamiento de $88.59\%$ y una precisión de $0.8938\%$ con el set de validación, a las que se les puede asignar distintos espacios de tiempo en que se realizan.

Este generador permite fabricar secuencias tan largo como se quiera y tantas como se desee. Solo se debe interactuar con el modelo provisto y previamente entrenado.

#### Anexos

Proceso de aprendizaje del modelo actual:

![m3](/media/abelo/Local Disk/4to/Mio/2do Semestre/MO II/project/m3.png)

Proceso de aprendizaje de un modelo con la misma arquitectura, pero utilizando $rnn_u = 64$ y entrenando durante $epochs = 55$



![model used](/media/abelo/Local Disk/4to/Mio/2do Semestre/MO II/project/model used.png)

Modelo con mismos hiperparámetros que el empleado pero con arquitectura de dos $simple\_RNN$. Se obtiene una precisión de $87.23\%$ y tarda más.(Posee más parámetros a entrenar)

![rnn](/media/abelo/Local Disk/4to/Mio/2do Semestre/MO II/project/rnn.png)

