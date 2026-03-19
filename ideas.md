# Ideas de Mejora

## Sombra Compartida (Convoy Detection)
¿Te gustaría que añadiéramos una pequeña lógica de "Sombra Compartida", donde si el objeto B está muy cerca de A, ambos se ayuden mutuamente a confirmarse como sólidos? Sería el siguiente paso para detectar convoyes de vehículos.

### Visualización sombras
. La Etapa 1 limpia el ruido basándose en esta física, y la Etapa 2 lo confirma mediante la consistencia angular.

¿Te gustaría que implementáramos una visualización en RViz que dibuje una línea desde el sensor hasta el final de la sombra para que puedas verificar visualmente cómo cambian los triángulos según la altura del objeto?