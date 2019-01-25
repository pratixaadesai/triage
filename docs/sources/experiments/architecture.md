# Experiment Architecture

This document is aimed at people wishing to contribute to Triage development. It explains the design and architecture of the Experiment class.

## Class Design

The Experiment class is designed to have all work done by component objects that reside as attributes on the instance. The purpose of this is to maximize the reuse potential of the components outside of the Experiment, as well as avoid excessive class inheritance within the Experiment.

The inheritance tree of the Experiment is reserved for *execution concerns*, such as switching between singlethreaded, multiprocess, or cluster execution. To enable these different execution contexts without excessive duplicated code, the components that cover computationally or memory-intensive work generally implement methods to generate a collection of serializable `tasks` to perform later, on either that same object or perhaps another one running in another process or machine.  The subclasses of Experiment then differentiate themselves by implementing methods to execute a collection of these `tasks` using their preferred method of execution, whether it be a simple loop, a process pool, or a cluster.

The components are created and experiment configuration is bound to them at Experiment construction time, so that the instance methods can have concise call signatures.

## Storage Abstractions

Another important part of enabling different execution contexts is being able to pass large, persisted objects (e.g. matrices or models) by reference to another process or cluster. To achieve this, as well as provide the ability to configure different storage mediums (e.g. S3) and formats (e,g, HDF) without changes to the Experiment class, all references to these large objects within any components are handled through an abstraction layer.

### Matrix Storage

All interactions with individual matrices and their bundled metadata are handled through `MatrixStore` objects.  The storage medium is handled through a base `Store` object that is an attribute of the `MatrixStore`. The storage format is handled through inheritance on the `MatrixStore`: Each subclass, such as `CSVMatrixStore` or `HDFMatrixStore`, implements the necessary methods (`save`, `load`, `head_of_matrix`) to properly persist or load a matrix from its storage.

In addition, the `MatrixStore` provides a variety of methods to retrieve data from either the base matrix itself or its metadata. For instance:

- `matrix` - the raw matrix
- `metadata` - the raw metadata dictionary
- `exists` - whether or not it exists in storage
- `columns` - the column list
- `labels` - the label column
- `uuid` - the matrix's UUID
- `as_of_dates` - the matrix's list of as-of-dates
- `matrix_with_sorted_columns` - the matrix' columns in an order given by the caller

One `MatrixStorageEngine` exists at the Experiment level, and roughly corresponds with a directory wherever matrices are stored. Its only interface is to provide a `MatrixStore` object given a matrix UUID.

### Model Storage

Model storage is handled similarly to matrix storage, although the interactions with it are far simpler. One `ModelStorageEngine` exists at the Experiment level, configured with the Experiment's storage medium, and through it trained models can be saved or loaded. The `ModelStorageEngine` uses joblib to save and load compressed pickles of the model.

### Miscellaneous Project Storage

Both the `ModelStorageEngine` and `MatrixStorageEngine` are based on a more general storage abstraction that is suitable for any other auxiliary objects (e.g. graph images) that need to be stored. That is the `ProjectStorage` object, which roughly corresponds to a directory on some storage medium where we store everything. One of these exists as an Experiment attribute, and its interface `.get_store` can be used to persist or load whatever is needed.

## Components

These are where the interesting data science work is done.

- [Timechop](#timechop) (temporal cross-validation)
- Architect (design matrix creation)
    - [Cohort Table Generator](#cohort-table-generator)
    - [Label Generator](#label-generator)
    - [Feature Generator](#feature-generator)
    - [Feature Dictionary Creator](#feature-dictionary-creator)
    - [Feature Group Creator](#feature-group-creator)
    - [Feature Group Mixer](#feature-group-mixer)
    - [Planner](#planner)
    - [Matrix Builder](#matrix-builder)
- Catwalk (modeling)
    - [Model Train/Tester](#model-train-tester)
    - [Model Trainer](#model-trainer)
    - [Predictor](#predictor)
    - [Model Evaluator](#model-evaluator)
    - [Individual Importance Calculator](#individual-importance-calculator)

### Timechop

#### Input

- `temporal_config` in experiment config

#### Output

- Time splits containing temporal cross-validation definition, including each `as_of_date` to be included in the matrices in each time split


### Cohort Table Generator

#### Input

- All unique `as_of_dates` needed by matrices in the experiment, as provided by [Timechop](#timechop)
- query and name from `cohort_config` in experiment config

#### Output

- A cohort table in the database, consisting of entity ids and dates


### Label Generator

#### Input

- All unique `as_of_dates` and `label_timespans`, needed by matrices in the experiment, as provided by [Timechop](#timechop)
- query and name from `label_config` in experiment config

#### Output

- A labels table in the database, consisting of entity ids, dates, and boolean labels


### Feature Generator

#### Input

- All unique `as_of_dates` needed by matrices in the experiment, and the start time for features, as provided by [Timechop](#timechop)
- The populated cohort table, as provided by [Cohort Table Generator](#cohort-table-generator)
- `feature_aggregations` in experiment config

#### Output

- Populated feature tables in the database, one for each `feature_aggregation`


### Feature Dictionary Creator

#### Input

- Names of feature tables and the index of each table, as provided by [Feature Generator](#feature-generator)

#### Output

- A master feature dictionary, consisting of each populated feature table and all of its feature column names.


### Feature Group Creator

#### Input

- Master feature dictionary, as provided by [Feature Dictionary Creator](#feature-dictionary-creator)
- `feature_group_definition` in experiment config

#### Output

- List of feature dictionaries, each representing one feature group

### Feature Group Mixer

#### Input

- List of feature dictionaries, as provided by [Feature Group Creator](#feature-group-creator)
- `feature_group_strategies` in experiment config

#### Output

- List of feature dictionaries, each representing one or more feature groups.

### Planner

#### Input

- List of feature dictionaries, as provided by [Feature Group Mixer](#feature-group-mixer)
- List of matrix split definitions, as provided by [Timechop](#timechop)
- `user_metadata`, in experiment config
- `feature_start_time` from `temporal_config` in experiment config
- cohort name from `cohort_config` in experiment config
- label name from `cohort_config` in experiment config

#### Output

- List of serializable matrix build tasks, consisting of everything needed to build a single matrix:
    - list of as-of-dates
    - a label name
    - a label type
    - a feature dictionary
    - matrix uuid
    - matrix metadata
    - matrix type (train or test)

### Matrix Builder

#### Input

- A matrix build task, as provided by [Planner](#planner)
- `include_missing_labels_in_train_as` from `label_config` in experiment config
- The experiment's [MatrixStorageEngine](#matrix-storage)

#### Output

- The built matrix saved in the [MatrixStorageEngine](#matrix-storage)
- A row describing the matrix saved in the database's `model_metadata.matrices` table.

### ModelTrainTester

A meta-component of sorts. Encompasses all of the other catwalk components.

#### Input

- One temporal split, as provided by [Timechop](#timechop)
- `grid_config` in experiment config
- Fully configured [ModelTrainer](#model-trainer), [Predictor](#predictor), [ModelEvaluator](#model-evaluator), [Individual Importance Calculator](#individual-importance-calculator) objects

#### Output

- All of its components are run, resulting in trained models, predictions, evaluation metrics, and individual importances

### ModelTrainer

#### Input
#### Output

### Predictor

#### Input
#### Output

### ModelEvaluator

#### Input
#### Output

### Individual Importance Calculator

#### Input
#### Output
