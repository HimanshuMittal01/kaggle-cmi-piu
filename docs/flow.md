```mermaid
flowchart TD
    subgraph preprocessing
    A[Start] --> B[Preprocessing Stream 1]
    A[Start] --> C[Preprocessing Stream 2]
    B --> D[Join Data]
    C --> D[Join Data]
    end
    subgraph feature_engineering
    D --> E[Split Feature Sets]
    E --> F[Feature Set 1] --> I[Collect Feature Sets]
    E --> G[Feature Set 2] --> I[Collect Feature Sets]
    E --> H[Feature Set 3] --> I[Collect Feature Sets]
    I --> J[End]
    end
```

```mermaid
flowchart TD
    A[Start] --> Evaluate1 -.- B[Train Model 1] --> E[End]
    A[Start] --> Evaluate2 -.- C[Train Model 2] --> E[End]
    A[Start] --> Evaluate3 -.- D[Train Model 3] --> E[End]
```

It is recommended to define which dataset version(s) for the raw input are used in the `start` step.
If storage/memory footprint is too high, then serialize and deserialize artifacts manually between the steps instead of using `.self` for versioning.

Dataset | Train/Valid
