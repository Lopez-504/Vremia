11-08-25 
Github: [link](https://github.com/graemecan/programacion_gpu) 
HTMLs: 
	--> Introducción: [link](https://graemecan.github.io/programacion_gpu/introduccion/introduccion.html#/1) 
	--> Memoria: [link](https://graemecan.github.io/programacion_gpu/memoria/memoria.html#/)
	--> Thread: [link](https://graemecan.github.io/programacion_gpu/threads/threads.html#/)	
CUDA doc: [link](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)

- Usaremos C y una extensión de C llamada CUDA, la cual es necesaria para lanzar trabajos 
	- También podemos utilizar CUDA a través de Python
- Evaluaciones: (Actualizado Aug-28)
	1. Prueba 30%
	2. Proyecto: informe 60% (3-4 páginas, formato paper) + presentación 40%
- El uso de GPU en librerías como PyTorch o TensoFlow es practicamente trivial, no es necesario programar explícitamente la parte relacionada a la GPU
- Programa:
	- Intro a CUDA
	- Uso de la memoria del GPU
	- Control de threads
	- Invocación de *kernels*
	- Librerías de CUDA y Python

# Intro a CUDA 

 - El GPU surge como una unidad de procesamiento gráfico, pero luego se comienza a utilizar para otro tipo de tareas. Lo que hace bien la GPU son **tarea simples**, pero **repetitivas**, muy adecuado para la paralelización
 - NVIDIA desarrolla un lenguaje para utilizar la GPU en tareas más generales
	 - GPGPU: General Porpouse GPU
- Programación heterogénea: este es el paradigma que envuelve a la programación en GPU, ya que en general entendemos al GPU como una **arquitectura aparte**.
	- Tenemos una conexión física entre la DRAM de la CPU y la DRAM de la GPU. Luego, para utilizar la GPU, debemos **transferir datos** entre ambos componentes
- En GPU tenemos en general 2 ordenes de magnitud más núcleos que en un CPU de un laptop común
- ==**Compute capability**==: escalar para indicar el nivel de cómputo de la GPU. Este escalar es similar a la versión de CUDA que debemos utilizar para esa GPU
- CPU vs GPU: 
	- CPU diseñado para cómputo secuencial, la GPU para cómputo paralelo
	- Una idea sencilla podría ser utilizar la GPU para la parte más intesiva de cálculos de nuestro programa y la parte secuencial con CPU
	- El paralelismo del GPU también funciona con threads (al igual que la CPU) pero en el caso del GPU, los thread son más **livianos y flexibles**. El *context switching* es mucho más rápido que en el CPU
		- This *context switching* is the process by which the CPU or GPU stops executing one task (process or thread) and resumes another, by saving the current state (context) of the first task and loading the saved state of the second.
	- La CPU busca minimizar la **latencia**, el GPU busca maximizar el *thoughpup* (algo así como la cantidad de datos que pasan por el GPU/CPU, por cada segundo)
- NVCC compiler: 
							![[Pasted image 20250824160703.png]]
- Código del *host* -> CPU
- Código del *device* -> GPU
- La convención es usar la extensión `.cu` para los programas escritos en CUDA

## Hola Mundo

```c
#include<stdio.h>
#include<stdlib.h>

__global__ void imprimir_del_gpu() {
	printf("Hola Mundo! desde el thread [%d,%d] del device\n", threadIdx.x,blockIdx.x);
}

int main() {
	printf("Hola Mundo desde el host!\n");
	imprimir_del_gpu<<<1,10>>>();         //blocks, threads per block 
	cudaDeviceSynchronize();              //Synchronize CPU and GPU
	return 0;
}
```
- Las funciones que tienen `__global__` pueden correr tanto desde el CPU (*host*) como en el GPU (*device*)
	- A esta función le llamamos **_kernel**
- El compilador NVCC ya incluye el header para poder usar la GPU, por eso no necesitamos hacer un `#include` para usarla.
- `cudaDeviceSynchronize()`: sincroniza la GPU con la CPU, esta última va a esperar a que la GPU termine su tarea para seguir con el resto del código.
- Usamos GPU T4 en Colab: ver notebook **HolaMundoGPU.ipynb**
	- Los threads están organizados en bloques. En CUDA, la variable `blockIdx.x` indica el indice del bloque, y la variable `threadIdx.x` indica el índice del thread dentro de cada bloque (desde 0 a n-1, con n thread por bloque)
	- Notación: <<<10,2>>>   -> number of blocks, number of threads per block
	- Notar que al correr el programa, no obtenemos **ningún error** si es que no se utiliza la GPU 
		- Ya que al ejecutar el kernel, este no nos devuelve un mensaje de error al CPU. Recuerda que los kernels son de tipo `void` 
		- Debemos obtenerlo nosotros usando `cudaGetErrorString(err)`
	- Para compilar, usar: `!nvcc -arch=sm_75 gpu_hello.cu -o a.out`
		- El 75 refiere al nivel de computo de este GPU Tesla 4.
- Para correr en kosmos, usar la maquina `flare` y carga el modulo `nvhpc/20.9`
	- La GPU que tenemos en `flare` también es una Tesla 4
- Los bloques corren de manera independiente, pero dentro de cada bloque, todos los thread ejecutan la misma instrucción en el mismo momento (**hay sincronización**). Notar que los threads aún están paralelizados
	- Por el diseño de la GPU, no es posible sincronizar los bloques, estos son **totalmente independientes**.
- Los kernels los lanzamos desde el CPU y estos corren paralelamente al CPU (al lado, recuerda que son 2 dispositivos independientes) 

**_Concepts**:
- _Thread_: secuencia de instrucciones, manejada por un **_scheduler_** (planificador, componente que reparte el tiempo disponible de un procesador entre los threads/procesos).
- _Context switching_: cambio de contexto de un _thread_, basicamente parar la operación de un _thread_ para permitir la operación de otro.
- _Latency_ (latencia): retraso entre la emisión de una instrucción y la transferencia de datos pedidos por la instrucción. 
- _Throughput_: la cantidad de datos que pasan a través de una red de comunicación en cierto unidad de tiempo (típicamente medido en GB/s)
- _Bandwidth_: el máximo teórico del _throughput_ para una red de comunicación.

---
14-08-25 | Lecture

 - Veamos cómo paralelizar lo que hace el programa `suma_vectores_host.c` 
 - Usamos el siguiente kernel: 
```c
__global__ void suma_device(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
    }
```

- Notar que en este caso no tenemos un bucle `for`
- Para identificar de manera unívoca cada thread, usamos 3 variables:
	- threadIdx.x: índice del thread respecto a su propio bloque
	- blockIdx.x
	- blockDim.x: cantidad de threads de cada bloque
	- Así, el índice será: idx = threadIdx.x + blockId.x * blockDim.x
- Todos los threads van a ejecutar esta operación de manera paralela. Notar que habrá sincronización en cada bloque, pero no entre bloques
- El `.x` refiere a la dimensión (x, y, z) en la que están esos threads/bloques  

- Antes de poder ejecutar estas operaciones, tenemos que **transferir los datos** desde la ram de la CPU al ram del GPU: todo con punteros
```c
	// Asignar memoria al lado del device (GPU)
	cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

	// Copiar al device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
```
- La variable `cudaMemcpyHostToDevice` es un número entero guardado en la librería de Cuda que nos permite especificar la **dirección** de la tranferencia de datos

- Luego, llamamos al kernel:
```c
    // Invocar kernel con 2 bloques de N/2 threads cada uno
	suma_device<<<2,N/2>>>(d_a,d_b,d_c);
    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n",cudaGetErrorString(err));

    // Copiar resultado al host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
```
- Con `cudaGetLastError()` y `cudaGetErrorString(err)` podemos capturar algún error generado a nivel de GPU, el cual no vemos por defecto, ya que no está en la ram del CPU.

- Siempre debemos copiar datos del CPU al GPU y una vez finalizado el trabajo, del GPU al CPU
	- Una desventaja de esto es que no podemos visualizar lo que está haciendo la GPU a menos que nos enviemos los datos de vuelta al CPU, lo cual podría disminuir bastante la eficiencia del cálculo.
	- Ahora, prácticamente todo lo que se puede hacer en el CPU se puede hacer en el GPU, por lo cual podríamos implementar alguna visualización de resultados parciales, pero esto agregaría más trabajo a la GPU
- La **velocidad de transferencia** de datos entre la CPU y GPU es prácticamente **independiente** de la dirección de la transferencia
- [x] Ejercicio: modificar este código cambiando el kernel para usar 2 bloques con 2048 thread por bloque
	- Esto nos genera un `invalid configuration argument`, ya que por la arquitectura de la GPU, `1024` es la cantidad máxima de threads por bloque. 
	- Para usar más threads, debemos agregar más bloques
- En este código no estamos imprimiendo desde la GPU, sino secuencialmente desde la CPU, por tanto los outputs están ordenados
	- Dentro de los bloques, los thread trabajan en grupos de 32. Estos grupos se denominan _**WARPS**_:
		- Trabajan de manera sincronizada y son la unidad de trabajo mínimo dentro de los bloques. 
		- Esto explica, por ejemplo, que en nuestro output veamos bloques de 32 lineas que se imprimen de manera **ordenada**
- En lo posible, queremos **minimizar la transferencia** de datos entre la CPU y GPU, ya que esta es mucho más lenta que el resto de operaciones
	- Este es el objetivo general de la optimización, queremos que nuestro código sea *compute bound* no *memory bound*

---
## Kernels

- Son funciones que corren en el GPU, las invocamos con `kernel_name<<<N,M>>>(...)`, pero tienen ciertas restricciones:
	- Solo tiene acceso a la memoria del device
	- El tipo de retorno debe ser `void` (porque no tenemos acceso al CPU)
	- No se pueden usar variables estáticas ni punteros a funciones
	- Corren **asincrónicamente**: una vez que comience a correrse esta parte de nuestro código, el resto va a seguir ejecutándose
		- Luego, cuando requerimos sincronización entre la GPU y le CPU, debemos explicitarla: `cudaDeviceSynchronize()`
- Los kernels están basados en SPMP (_single program multiple data_)
- Un _kernel_ corresponde a **código escalar** para un solo _thread_.
- Al invocar el _kernel_ muchos _threads_ estarán realizando la misma operación, definida en el _kernel_.
## Organización de los threads

- Estos se pueden organizar lógicamente en 1, 2 o 3 dimensiones, dependiendo de lo que resulte más conveniente a nuestro problema
- Esto explica porqué hemos estado usando `.x` cuando nos referimos a los índices de los bloques y threads. También tenemos `.y` and `.z` para más dimensiones
- El grupo entero de threads corriendo en el GPU se denomina `grid`. Este grid se compone de `blocks` y estos por `threads`
	- Todos los threads del `grid` tienen acceso a la `memoria global`
	- Cada `block` tiene un espacio de `memoria compartida`
- Luego al kernel le podemos pasar variables tipo `dim3`
```c
dim3 bloques (bx,by,bz); 
dim3 grid (gx,gy,gz); 
nombre_kernel<<< grid, bloques >>>(...)
```
	Sin embargo, en general solemos usar solamente una dimensión para organizar nuestros threads
- Independiente del número de dimensiones, la cantidad máxima de threads sigue siendo 1024
- Ver ejemplo en colab: [mostrarIndices.cu](https://colab.research.google.com/drive/1D3s-3aAejkb-72IS1nsqGsAw5WKpednM#scrollTo=YkQdkJW5JIvH)
	- Dentro de los bloques hay orden (sincronización), pero fuera de ellos no, ya que los bloques correr de manera **independiente**.

### Warps, bloques y grids

- Los threads en un warp están **sincronizados implícitamente**
- Todos los threads en un bloque tienen acceso a un espacio de **memoria compartida**, pero dado que los bloques son independientes, no hay comunicación entre threads de distintos bloques

## Funciones en CUDA

- `__global__`: ejecuta en el _device_, se puede llamar desde el _host_ y el _device_ (para _compute capability_ mayor que 3).
- `__host__`: ejecuta en el _host_, se puede llamar desde el _host_ (típicamente no es necesario especificar una función de esta manera)
- `__device__`: ejecuta en el _device_, se puede llamar solo desde el _device_.
- Se puede compilar una función tanto para el _host_ como para el _device_ combinando `__host__` y `__device__`.

## Manage errors

- Todas las funciones del API de CUDA devuelven un número (un _enum_) que corresponde a algún tipo de error, y desde este número podemos obtener un mensaje de error:
```c
cudaError_t err = cudaMemcpy(...);
cudaGetErrorString(err);
```

- Una mejor forma es usar una macro:
```
#define CHECK(llamada) \ { \ const cudaError_t err = call; \ if (err != cudaSuccess) \ { \ printf("Error: %s:%d, ", __FILE__, __LINE__); \ printf("codigo de error:%d, mensaje: %s\n", err, cudaGetErrorString(err)); \ exit(1); \ } \ }
```

- Further documentation: [link](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html)

## Profiling

- Profiling refiere a testear qué tan eficiente es nuestro código
- Tenemos programas llamados *profilers* que entregan información acerca de la ejecución de un código (tiempo de ejecución, utilización de memoria, etc)
	- Luego, en lugar de intentar optimizar todo nuestro código al pasar de una versión secuencial a una paralela, mejor hacemos un profiling y nos enfocamos en paralelizar las partes menos eficiente de nuestro código, i.e., bottleneck.
	- Para CUDA tenemos `nvprof` (antiguo, uso en terminal), `ncu` y `nsys` 

- [ ] PROYECTO: Enviar propuesta (HTML) explicando qué queremos hacer y todas las formalidades de una propuesta
	- Abstract, propósito, metodología, resultados esperados
	- 1 página

---
18-08-25 | Lecture 

- Cuando el profiler testea el código, lo corre varias veces y nos da tiempos mínimo, máximo y promedio.
- En este profile aparecen todos los procesos que se llevan a cabo durante la ejecución del código
- Para GPUs modernas (capacidad de cómputo $\geq$ 7) tenemos los profilers *NSight compute* y *NSight system*
	- Estas tienen una interfaz visual
	- La segunda nos muestra un timeline para nuestro código, incluyendo los tiempo del CPU y GPU para cada operación
	- Ambas utilizan **métricas**, i.e., información bastante detallada del uso del GPU (e.g., uso de cada thread)
- El uso más básico de un profiler es distinguir si acaso nuestro código es **compute bound** (caso ideal) o **memory bound** (quizás puede mejorar al optimizarlo de otra forma)
	- _Compute bound_: el rendimiento del programa está limitado por la rapidez de las operaciones aritméticas/matemáticas del GPU.
	- _Memory bound_: el rendimiento del programa está limitado por la rapidez de la comunicación con la memoria del GPU.
- Ver código [simpleDeviceQuery.cu](https://colab.research.google.com/drive/1D3s-3aAejkb-72IS1nsqGsAw5WKpednM#scrollTo=5y9mQXvlpkDp) en colab

## Memoria

- Jerarquía en la CPU: (de más rápido a más lento) registros, caches, main memory, disk memory. Los más rápidos también son los más pequeños.
- Jerarquía en la GPU: memoria global (todos los threads pueden acceder a ella), shared memory (para cada bloque), y más! 
						![[Pasted image 20250824204616.png]]
	- Los bloques de memoria más relevantes son la **memoria global** y la **memoria compartida**
		- La memoria principal tiene una latencia alta y un bajo ancho de banda. Recordar que lo más importante es el **_throughput** 
	- La *memoria constante* mantiene sus valores durante todo el cálculo, las copiamos desde la CPU y así se quedan, podemos leer sus datos con una velocidad un poco mayor a la memoria global o compartida. Uso, por ejemplo, constantes físicas 
### Global Memory
- El uso eficiente de esta memoria es muy importante para optimizar un código de CUDA
	- No hay mucha diferencia entre la eficiencia de una declaración estática y dinámica
- Para acceder a sus datos de manera eficiente, debemos usar un acceso **lineal** y **contiguo** Notar que en este acceso también tenemos un caché 
	- **Alineado**: algo técnico relacionado al tamaño de cada elemento
	- **Contiguo**: muy similar al caché, queremos que cada thread acceda a un espacio en memoria contiguo al espacio accedido anteriormente 
- Ver códigos copiarFila.cu y copiarColumna.cu
	- Recordar que para definir de manera más eficiente, usamos **bitshift**. Por ejemplo, para definir el número entero $2^{11}$, hacemos
```c
		int nx = 1 << 11;
```

- Cuando lanzamos algún kernel por primera vez en nuestro código, este va a demorar un poco más de lo normal, ya que el sistema debe **inicializar el uso de la GPU**
- **_Métricas_**: en ncu tenemos métricas que miden la velocidad de cargar/guardar datos
- Al cargar datos (*load*) las operaciones pasan por un caché y por tanto son más eficientes que las operaciones de *store*, donde no tenemos caché.
	- Si podemos al menos guardar los datos de manera contigua, entonces ganaremos algo de eficiencia. Por ejemplo, cargar por columnas y guardar por filas 
- Tenemos 2 opciones para estructurar nuestros datos: AoS (array of structures) y SoA (structure of arrays)
	- Selecionamos una de estas dependiendo del problema que estemos resolviendo

### Shared Memory
- Es **más rápida** que la memoria global, pero está limitada a cada bloque 
- Esta es *on.chip*, por tanto tiene **bandwidth alto y baja latencia**
- Para declarar variables de manera **estática** usamos `__shared__ float tile[ny][nx];` (se llaman tiles porque son como baldosas)
- Y de manera **dinámica**: `extern __shared__ int tile[];` Notar que solo puede ser unidimensional
- Una métrica importante es la **_ocupación_** 
- Una estructura básica sería utilizar un tamaño de la memoria compartida igual al tamaño del bloque
- Al invocar al kernel, le pasamos el **tamaño de la memoria compartida**
- Ejemplo: traspuesta de una matriz. Ver código [transpuesta.cu](https://colab.research.google.com/drive/1D3s-3aAejkb-72IS1nsqGsAw5WKpednM#scrollTo=TFPkGpKmzbHM) 
	- Necesitamos índices globales y locales (para la memoria compartida)
	- Notar que siempre debemos tener cuidado con la sincronización. Usamos `__synchtreads` para sincronizar los threads dentro de un bloque, los threads en **diferentes bloques no se sincronizan**
- La memoria compartida se organiza en ***banks***:
	- Si cada thread accede a un banco de manera separada, todo bien
	- Si 2 o más threads intentan acceder al mismo elemento del banco, tampoco hay problema, solo estamos leyendo el mismo valor
	- Pero si tenemos 2 o más threads accediendo a distintas filas en un mismo banco, tendremos un acceso secuencial, menos eficiente
		- Para resolver esto, llamado **conflicto de bancos**, usamos **padding**. Esto es, como siempre, añadir espacios vacíos al final de las estructuras para que los elementos se distribuyan de tal manera que el acceso a la memoria sea más eficiente
### Constant memory
- Este tine un caché asociado, por tanto es más rápido acceder a ella
- Debe tener un *scope* global, esto es, debe estar disponible incluso fuera de cualquier kernel
- 64kb
- Útil para guardar constantes matemáticas
	- Por ejemplo, en el código *memoria_compartida.cu*
- En ocasiones es útil, pero siempre podemos trabajar con otros tipos de memoria
### Memoria unificada
- En el *host* tenemos una memoria *pageable*, esto es, la memoria está organizada en paǵinas que el SO puede mover a la memoria virtual (disco duro)
	- Cuando el sistema requiere datos del disco duro, ocurre un *page fault* y se trae una página desde el disco duro a la ram 
	- En ocasiones no podemos transferir datos de la ram al disco duro, ya que esta memoria está fijada (pinned memory)
	- El GPU no tiene control sobre todas estas cosas, luego los datos se transfieren de _pageable_ a _pinned_ (en el host) y después al _device_
	- Con `cudaMalloc` asignamos memoria en la GPU, pero con `CudaMallocHost` podemos crear *pinned memory*
		- Obs: esto no es recomendable, ya que puede resultar en una caída del sistema
- Ahora, la memoria unificada nos permite acceder de igual forma desde la CPU o GPU, aún hay transferencia de datos, pero esta se hace automáticamente. Por tanto, debemos tener cuidado en la forma en que usamos esta memoria, ya que muchas transferencias van a ralentizar nuestro código.
	- Podemos asignar memoria de manera estática o dinámica: ver funciones de CUDA
	- Basta con declarar un solo array para poder acceder a él tanto desde el CPU y como el GPU.
	- ver código: memoria_unificada.cu
	- **Note**: keep track of data transfers, if u get too used to using unified memory, u'll most likely run into some data transfer bottleneck
## Programación de *threads*

>[!Los cores de un Gpu son más lentos que los de un CPU, pero tienen un context swtching mucho más rápido y además en la GPU tenemos muchos más threads que en la CPU] 

- Tenemos: grid -> bloques -> warps -> threads
- Dentro de un bloque, podemos sincronizar los threads usando `__syncthreads()`. Y en los warps tenemos una **sincronización implícita**
- En CUDA tenemos Single Instruction Multiple Threads, un modelo híbrido entre **SIMD** y **SMT** (simultaneous multithreading). Esto es ligeramente diferente a lo que teníamos en programación paralela en CPU
	- Para la GPU tenemos menos instrucciones posibles, pero estas son más **flexibles** y **simples**, menos poderasas que las del CPU.
	- En general no queremos introducir un ***warp divergence***: 
		- Cuando dentro de un warp tengo algunos threads ejecutando una instrcucción y otros ejecutando otra instrucción
- **Ocuppancy**: warps activos / max warps activos. 
	- Esta es una de las cantidades más importantes para la eficiencia en GPU
	- Cuando la carga de cada thread es muy alta, nuestro ocuppancy disminuye
- **Registros**: unidad de memoria pequeña y rápida (ver figura memoria GPU)
	- El compilador calcula la cantidad de registros que necesita en base a las variables locales que estemos usando
	- La cantidad de registros puede introducir una limitación para el ocuppancy

- Respecto a la memoria compartida, esta también introduce una limitación en el número de threads que podemos tener corriendo de manera paralela.
- También tenemos una limitación en el tamaño de nuestros bloques
- Todas estas cosas afectan el *occupancy* 

- Si nuestro bandwidth es pequeño, una buena idea es intentar mejorar nuestro *occupancy*
- Un occupancy de 1 indica que estamos aprovechando de buena forma los recursos de la GPU
- En ocasiones el compilador no podrá darnos lo que le pedimos, en este caso se genera un *memory spill* que ralentiza nuestro código
	- Luego, queremos limitar la cantidad de variables locales que requieren nuestros kernels, ya que a **más variables locales, más registros**. Ligado al diseño de nuestro código

- **Reducción**: cuando a partir de un conjunto queremos obtener un solo valor, por ejemplo calcular un promedio, de forma **paralela**.
	- En el caso del GPU, debemos programar la reducción, no como en MPI
	- Ver diagramas con la forma de estructurar la reducción en GPU
	- Ver código [[Optimizacion Reduccion Global]]
		- ***stride***: salto en los elementos del array, en el caso de este código, lo usamos para hacer saltos en los indice de los threads. 
		- Este código, con saltos `*=2`, hacemos justamente lo que describe el diagrama
											![[Pasted image 20250821112201.png]]
		- Uno de los problemas de este código es que hace muchas sumas innecesarias, a saber, todas las de los espacio intermedios (color blanco)
	- Estos código son bastante buenos para estudiarlos. 
		- Ver código [[Optimizacion Reduccion Global]] donde incluyo todas las versiones, desde la menos a la más optimizada. 
		- Quizás todo este texto puede reducirse y añadirse mejor en [[Optimizacion Reduccion Global]]

	- Dado que los warps están sincronizados, usar un `if else` va a introducir un cálculo secuencial y por tanto un uso ineficiente de la GPU ya que algunos threads van a estar esperando a los otros.
	- Desde el github obtener las métricas (en este caso ver la occupancy) que queremos utilizar en el profiler (ver colab)
		- metrics names here...
	- Para poder reemplazar el `if else` y no tener *warp divergence*, podemos multiplicar el índice de los threads por 2 y por el stride. Ver [[Optimizacion Reduccion Global]]
	- Otra opción para implementar la reducción (reduccion_global4.cu) es hacer la mayoría de las reducciones en el GPU y la reducción final en el host
		- En este caso solamente estamos llamando el kernel una vez
		- Es necesario hacer una sincronización explícita a nivel del bloque
		- Pero este aún tiene el problema de la memoria: **no estamos accediendo de manera contigua** 
	- Para solucionar el acceso contiguo (reduccion_global5.cu) partimos con un stride grande y vamos guardando en lugares más cercanos, esto nos permite

	- Una última optimización para este código (reduccion_global6.cu) es usar **loop unrolling**. Esto es, hacer menos iteración, pero más trabajo por iteración (esto es más eficiente porque cada iteración del `for` tiene un coste intrínseco por el hecho de confirmar que el `for` debe seguir siendo ejecutado)
go from this
```c
		for (int i = 0; i < N; i++){ 
			a[i] = b[i] + c[i]; }
```
to this
```c
		for (int i = 0; i < N; i+=2){ 
		    a[i] = b[i] + c[i];
		    a[i+1] = b[i+1] + c[i+1]; }
```
or even this
```c
		for (int i = 0; i < N; i+=4){ 
		    a[i] = b[i] + c[i];
		    a[i+1] = b[i+1] + c[i+1];
		    a[i+2] = b[i+2] + c[i+2];
		    a[i+3] = b[i+3] + c[i+3]; }
```

- Usamos el loop unrolling antes de hacer la reducción
- Esta optimización nos permite también reducir la cantidad de bloques necesarios, potencialmente mejorando el *occupancy*
- Así, podemos usar grupos de 4 u 8 bloques para hacer loop unrolling
- También podemos aplicar reducción paralela a nivel de warps: *warp unrolling* (pero no es tan popular)
- También tenemos *reducción compartida*, pero este tiene el contra de que tenemos que copiar datos
- Hay **muchas formas** de optimizar, **debemos probar**.

- El resultado de la GPU no es exactamente igual al de la CPU, en general, los errores se suelen acumular mucho más en códigos secuenciales que en reducciones paralelas, ya que se el error es mayor cuando sumamos números de disntinto orden de magnitud. Luego si suponemos que todos nuestros número son de order similar, al sumarlos todos menos 1, estaríamos en la última iteración sumando un número muy grande con uno pequeño (esto en el caso secuencial), pero si hacemos reducción paralela podemos evitar esto
	- Cuando trabajamos con ANN en GPU, no solemos utilizar una precisión muy grande
- Revisar `pycuda`

---
23-08-25 | Study session

- Read *AMReX: Block-structured adaptive mesh reﬁnement for multiphysics applications* to get some ideas for my project. Try to link this with [[GPU with Einstein Toolkit]]

---
25-08-25 | Lecture

- Para pasar del cálculo secuencial al kernel, cambiamos el for por un cálculo de una sola iteración que va a correr en cada 
	- En general, si estamos haciendo operaciones con vectores de N elementos, usamos N threads 
	- Pero si N es muy grande, quizás sea conveniente usar menos threads y agregar un `for` dentro del kernel para compensar por el uso de menos threads.
		- Esto es útil para tener un buen *occupancy*, i.e., utilizar nuestros threads de manera eficiente
- Warp primitives: funciones que operan directamente con los threads dentro de un warp
	- Estas funcionan a nivel de registros (muy rápidos y pequeños) por tanto la manipulación de memoria es bastante eficiente
	- Ver documentación (jerga: lane, ) y códigos *warp_shuffle_down.cu* y otros 
	- Un *mask* corresponde a los threads que deben estar trabajando de manera convergente, esto es, en el mismo cálculo
	- En general, estas son complicadas de aplicar en comparación con los beneficios de optimización que entregan
	- Otra opción es usar *cooperative groups* (fuera del scope del curso, C++)

## Ejecución de kernels

- Un *stream* es una secuencia de comandos para el GPU
- Podemos especificar el *stream* al lanzar el kernel: es stream por defecto es el `0`
	- `foo_kernel<<<grid_size, block_size, shared_mem, stream>>>(...)`
- Ver ejemplo cuda_default_stream.cu, cuda_multi_stream.cu, cuda_multi_stream_with_sync (organizar en un solo archivo como hicimos antes) 
- Una idea puede ser lanzar distintos kernels en cada stream, estos corren de manera asincrónica, pero si queremos sincronización podemos hacer `cudaStreamSynchronize(stream)`. Pero notar que esto no sincroniza los streams entre ellos, sino con el host.
- Todos los streams son sincrónicos con el default stream, luego si queremos paralelizar a nivel de streams, no usar el default (`0`)
- Una aplicación relevante de los streams es paralelizar la tranferencia de datos con el cálculo (ver diagrama)
					![[Pasted image 20250825110431.png]]
	
	- Mientras corremos un kernel en un strema, usamos otro stream para hacer los cálculos de otro kernel
	- Para implementar esto debemos transferir datos de manera sincrónica, y para ello usamos la función `cudaMemcpyAsynch` en lugar de `cudaMemcpy`, la cual imponía una sincronización implícita entre el CPU y el GPU
		- Además, en el *host* debemos usar una memoria de tipo *pinned*, `cudaMallocHost()`. Esto dado que la transferencia de datos será asincrónica. 
		- Las operaciones de CUDA deben estar en streams diferentes y no usar el default stream
- Para obtener el status de un cierto stream
	- Ver ejemplo cuda_callback.cu
	- El uso de callbacks está obsoleto. Una alternativa es usar `cudaLaunchHostFun`, pero ya veremos una mejor forma: **cuda events**
- Podemos asociar prioridad a los streams
	- Al lanzarlos, no hay orden, por tanto si necesitamos que uno esté listo antes que otro, debemos especificar la prioridad
- Para grabar evento en el device usamos `cudaEvent`
	- Ver [[cuda_event.cu]]
	- Creamos variables de eventos
	- Los podemos usar para sincronizar la ejecución del kernel con el host
	- Ver [[cuda_event_with_streams.cu]] 
	- Es bastante flexible, podemos seleccionar cuál kernel queremos sincronizar con el host
## Syncronization

Sincronizar todo:
- `cudaDeviceSynchronize()`
- Bloquea host hasta que todas las instrucciones de CUDA terminen.

Sincronizar con respecto a un stream:
- `cudaStreamSynchronize( stream )`
- Bloquea host hasta que todas las instrucciones de CUDA en el stream terminen, pero los otros streams pueden seguir corriendo normalmente

Sincronizar con eventos
- crear eventos dentro de los streams para sincronizar
- cudaEventRecord( event,stream)
- cudaEventSynchronize(event)
- cudaStreamWaitEvent(stream, event): muy útil si uno de nuestros streams requiere que otro termine primero, solo debemos crear un evento al final del primer stream y el segundo va a esperar a este evento para poder comenzar
- cudaEventQuery(event): preguntar al GPU si acaso este evento ha ocurrido.

## CUDA Dynamic Parallelism

 - Lanzar kernels dentro de un kernel (recursividad)
	- Nos permite usar *child grids*, lo cual permite implementar grids adaptativos (muy necesarios en simulaciones, donde requerimos de aumentar la resolución en ciertas ocaciones)
	- Ver recursion.cu y 
	- CUDA soporta una recursividad máxima de 24
	- Los algoritmos recursivos son elegantes, pero costosos, recuerda que cada llamada a un kernel implica un costo

- Podemos usar conjuntamente CUDA/OpenMP
- También podemos combinar con MPI, pero debemos usar MPS (Multi-Process Service). Disponible solo para Linux. Es análogo a un *daemon* para poder ejecutar nuestros kernels de manera paralela 
	- A nivel del CPU, este se muestra como un solo proceso y el *daemon* se hace cargo del resto
	- Esto es bastante complicado, debes mezclar OpenMP, MPI y CUDA. En general, en lugar de mezclar, es mejor separar una parte del código para correrla en GPU y otra para el CPU

- Ejemplo cuda_kernel.cu
	- 3 formas de correr una operación SAXPY
	- Hay un costo cada vez que lanzamos un kernel, asociado a la inicialización de la GPU. Es mejor añadir un ciclo a un kernel que lanzar un kernel varias veces.
	- En el caso de la ecuación de onda, la mejor opción sería no lanzar un kernel para cada time-step sino
		- Esta una buena idea para mi proyecto. Cada cierta cantidad de iteraciones podemos sincronizar y sacar datos para hacer una posterior visualización
- Lo que ganamos al pasar del CPU al GPU es enorme en comparación con lo que ganamos con estas optimizaciones, por tanto en ocasiones bastará con pasar nuestro código a CUDA y ya está.

- Proxima clase escribimos código, ecuación de onda en el GPU y comparar con la versión secuencial
	- Con una grilla pequeña, la ganancia no será significativa, pero con grillas realistas o al menos grandes, vamos a notar que la GPU mejorar considerablemente la ejecución de nuestro código.
- Respecto al contenido del curso, nos quedan librerías y python para usar GPU
	- Con esto nos evitamos tener que escribir en CUDA, solo importamos librerías en C o Python (por ejemplo, JAX)

---
28-08-25 | Clase práctica

## Wave Eq 1D
$$\partial_t^2 u=c^2\partial_x^2 u$$

- Si utilizamos un método explícito, tener cuidado con la estabilidad. Manejar este con un CFL
- Si utilizamos un método implícito, es algo más complicado (intentar luego)

- Partir de un código secuencial (sencillo, condiciones de borde fijas y condición inicial con un seno) y luego pasar a GPU
- El ciclo sobre el tiempo no se podrá paralelizar (por la causalidad del problema) pero el ciclo sobre el espacio sí se puede paralelizar
1. En primer lugar, trabajar con memoria global o unificada
2. Luego, intentar usar memoria compartida
	- En este caso, debemos manejar las ghost zones ya que los bloques no tienen acceso a la memoria de otros bloques

- Traducción de CPU a GPU:
	- Cambiar malloc por cudaMalloc

- 1st try for waveEqCUDA.cu code
	- Q: do I need to initialize my variables in the CPU or can I just do this on the GPU
	- So, we need to do 2 things: update the interior and update the exterior, then we need 2 kernels
	- Q: How do we handle the constants? do we define them as macros and then their visible to both the CPU and GPU? Let's use global memory

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1000         // Number of spatial grid points
#define STEPS 1000     // Number of time steps
#define L 1.0          // Length of the domain
#define C 1.0          // Wave speed
#define CFL 0.9        // CFL number (must be <= 1 for stability)

//Definir kernel 
__global__ void update_interior(const double* __restrict__ u_prev,
                                const double* __restrict__ u,
                                double* __restrict__ u_next,
                                int n_points)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1; // start at 1
    if (i < n_points - 1) {
        u_next[i] = 2.0 * u[i] - u_prev[i]
                  + d_cfl2 * (u[i+1] - 2.0 * u[i] + u[i-1]);
    }
}

// Kernel para condiciones de borde (Dirichlet)
__global__ void apply_dirichlet(double* u_next, int n_points) {
    if (threadIdx.x == 0) {
        u_next[0] = 0.0;
        u_next[n_points - 1] = 0.0;
    }
}

int main() {
    double dx = L / (N - 1);
    double dt = CFL * dx / C; // Time step from CFL condition
    double cfl2 = (C * dt / dx) * (C * dt / dx);

    // Allocate arrays (CPU)
    double *u_prev = (double*)calloc(N, sizeof(double));
    double *u = (double*)calloc(N, sizeof(double));
    double *u_next = (double*)calloc(N, sizeof(double));

	// Initial condition: u(x,0) = sin(pi * x), du/dt = 0
    for (int i = 0; i < N; i++) {
        double x = i * dx;
        u[i] = sin(2 * M_PI * x);
        u_prev[i] = u[i]; // because du/dt = 0
    }

	// Asignar memoria al lado del device (GPU)
	double *d_u_prev, *d_u, *d_u_next;

	int size = N * sizeof(double);
    cudaMalloc((void **)&d_u_prev, size);
    cudaMalloc((void **)&d_u, size);
    cudaMalloc((void **)&d_u_next, size);

    // Copy from Host to Device
    cudaMemcpy(d_u_prev, u_prev, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_next, u_next, size, cudaMemcpyHostToDevice);

	// Invoke GPU kernel
	kernel_device<<<4,8>>>(d_u_prev,d_u,d_u_next);		
    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n",cudaGetErrorString(err));

    // Time evolution
    FILE *f = fopen("wave_output.csv", "w");	
    for (int n = 0; n < STEPS; n++) {
        for (int i = 1; i < N - 1; i++) {
            u_next[i] = 2*u[i] - u_prev[i] + cfl2 * (u[i+1] - 2*u[i] + u[i-1]);
        }

        // Apply Dirichlet boundary conditions: u=0 at both ends
        u_next[0] = 0.0;
        u_next[N-1] = 0.0;

	// Save current state to file		
	if (n % 50 == 0) {
            for (int i = 1; i < N; i++) {		
                fprintf(f, "%f", u[i]);
                if (i != N - 1) fprintf(f, ",");			
            }    
	    fprintf(f, "\n");
        }

        // Rotate pointers
        double* temp = u_prev;
        u_prev = u;
        u = u_next;
        u_next = temp;

        // Optionally output or visualize here
        // e.g., print u[N/2] every 10 steps
        //if (n % 20 == 0) {
        //    for (int i = 0; i < N; i++) {
        //        printf("%f ",u[i]);
        //    }
        //    printf("\n");
        //}
    }

	//CPU free
    free(u_prev);
    free(u);
    free(u_next);

	//GPU free

    return 0;
}  
```

Proxima semana no hay clases de GPU: Aug-01, Aug-04 and Aug-08
Prueba Oct-02 
Proyecto: 1 página (abstract, metodología, resultados esperados). Hasta Sep-05
	- Quizás hacer algo con *dense output*
	- La idea es que la parte de GPU lo programemos nosotros
Crear GitHub repo para subir tanto los códigos que yo escriba en clases como los del proyecto. Luego enviar link a Graeme.
