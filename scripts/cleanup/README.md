# Scripts históricos de limpieza

Estos scripts son evidencia del proceso inicial de ingestión y organización de
fuentes. Conservan referencias históricas a `data/clean/` y algunos ejecutan
movimientos, copias, renombres o eliminaciones directas.

No forman parte del pipeline activo y **no deben ejecutarse contra `data/` del
repositorio**. El corpus vigente vive fuera del repositorio en:

```text
$DATASET_ROOT/clean/
```

Los flujos activos deben resolver esa raíz mediante `get_dataset_root()` y
escribir datos derivados bajo `PROJECT_DATA_ROOT`. Estos archivos no se
reescribieron para no alterar la procedencia de las operaciones históricas.

Antes de reutilizar cualquier lógica:

1. extraerla a un módulo seguro;
2. reemplazar rutas hardcodeadas;
3. agregar modo dry-run y pruebas;
4. prohibir escrituras en la fuente activa;
5. revisar el target manualmente.
