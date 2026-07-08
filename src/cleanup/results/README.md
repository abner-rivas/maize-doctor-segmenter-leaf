# results/ - registro histórico de deduplicaciones

Los CSV de este directorio son la salida de `find_duplicates.py` de las deduplicaciones
**ya aplicadas** a `clean/` durante la curación del dataset. Se conservan a propósito como
registro histórico/auditable de qué duplicados se eliminaron y cuál copia se conservó.

Notas:

- Las rutas que contienen son absolutas y corresponden a la máquina donde se ejecutó la
  limpieza original (`/mnt/datasets/...`); no son reutilizables tal cual en otro entorno.
- No usarlos con la opción 3 de `find_duplicates.py` (eliminar desde CSV) salvo que las
  rutas sigan siendo válidas en tu máquina.
- Las nuevas ejecuciones de `find_duplicates.py` escriben aquí mismo, con timestamp en el
  nombre de archivo.
