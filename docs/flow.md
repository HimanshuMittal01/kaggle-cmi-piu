```mermaid
flowchart TD
    A[Start] --> B[Preprocessing Stream 1]
    A[Start] --> C[Preprocessing Stream 2]
    B --> D[Join Data]
    C --> D[Join Data]
    D --> E[Split Feature Sets]
    E --> F[Feature Set 1]
    E --> G[Feature Set 2]
    E --> H[Feature Set 3]
    F --> I[Collect Feature Sets]
    G --> I[Collect Feature Sets]
    H --> I[Collect Feature Sets]
    I --> J[End]
```

```mermaid
flowchart TD
    A[Start] --> B[Train Model 1] --> E[End]
    A[Start] --> C[Train Model 2] --> E[End]
    A[Start] --> D[Train Model 3] --> E[End]
```

It is recommended to define which dataset version(s) for the raw input are used in the `start` step.
If storage/memory footprint is too high, then serialize and deserialize artifacts manually between the steps instead of using `.self` for versioning.

Dataset | Train/Valid=