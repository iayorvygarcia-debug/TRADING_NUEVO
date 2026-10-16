# Auditoría de contabilidad y ejecución

## Hallazgos observados

Los datos del dashboard mostraban 20 operaciones con PnL realizado de `$0.00`, todas clasificadas como `WIN`, aunque el broker reportaba siete posiciones abiertas y una orden `NEW`. Esto no representaba operaciones cerradas reales.

También se observó el mensaje `Orden ejecutada` para una orden que todavía aparecía con estado `NEW`. Ese texto confundía la aceptación o envío de una orden con su ejecución efectiva.

Las posiciones Alpaca aparecían con `stop_loss` y `take_profit` nulos porque la telemetría solo buscaba las solicitudes internas del Paper Broker y no las patas bracket devueltas por Alpaca.

El valor de buying power era el reportado por el broker, pero la interfaz lo etiquetaba como margen fijo de 2×. Esa etiqueta no era válida para una cuenta Alpaca concreta y podía inducir a interpretar incorrectamente el apalancamiento disponible.

## Correcciones

El historial Alpaca ahora toma las órdenes llenadas y construye operaciones cerradas mediante emparejamiento FIFO por símbolo y dirección. Una orden llenada se considera una ejecución, no una operación cerrada. Solo se muestra una operación cuando una orden opuesta cierra una cantidad previamente abierta. El PnL se calcula con precio de entrada, precio de salida, cantidad y dirección; los resultados pueden ser `WIN`, `LOSS` o `BREAKEVEN`.

El feed del worker ahora distingue entre `Orden enviada ... pendiente de fill` y `Orden ejecutada`. El dashboard conserva esta diferencia en el registro de actividad.

La telemetría busca los niveles de las patas bracket activas en las órdenes Alpaca, de modo que los campos SL y TP muestran los niveles disponibles cuando el broker los devuelve.

La etiqueta de buying power fue cambiada a `BROKER REPORTED`, porque el valor debe interpretarse según el modelo de margen y las reglas de la cuenta configurada.

El dashboard ya no cuenta un PnL cero como ganancia. Los breakeven se muestran como `SIN PÉRDIDA` y no se suman artificialmente a ganadas o perdidas.

## Recomendación operativa

No debe habilitarse el auto-trading real basándose en el historial anterior del dashboard. Ese historial era una auditoría de órdenes llenadas, no un estado fiable de operaciones cerradas. Después de instalar esta versión, conviene reiniciar o revisar el historial mostrado y verificar que las nuevas operaciones se cierren mediante una orden opuesta o una pata bracket ejecutada.

La suite de validación posterior a los cambios contiene **205 pruebas aprobadas**.
