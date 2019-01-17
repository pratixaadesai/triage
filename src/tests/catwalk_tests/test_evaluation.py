from triage.component.catwalk.evaluation import ModelEvaluator, generate_binary_at_x
from triage.component.catwalk.metrics import Metric
import testing.postgresql
import datetime
import warnings

import factory
import numpy
import pandas
from sqlalchemy import create_engine
from triage.component.catwalk.db import ensure_db
from tests.utils import fake_labels, fake_trained_model, MockMatrixStore
from tests.results_tests.factories import ModelFactory, EvaluationFactory, init_engine, session
from tests.utils import fake_trained_model, MockMatrixStore
from tests.results_tests.factories import (
    ModelFactory,
    TestPredictionFactory,
    TrainPredictionFactory,
    SubsetFactory,
    init_engine,
    session,
)


@Metric(greater_is_better=True)
def always_half(predictions_proba, predictions_binary, labels, parameters):
    return 0.5


SUBSETS = [
    {
        "name": "evens",
        "query": """\
            select entity_id, as_of_date
            from {results_schema}.predictions
            where entity_id % 2 = 0
            and as_of_date in (SELECT(UNNEST(ARRAY{as_of_dates}::timestamp[])))
            and model_id = '{model_id}'
        """,
        "hash": "klmno",
    },
    {
        "name": "odds",
        "query": """\
            select entity_id, as_of_date
            from {results_schema}.predictions
            where entity_id % 2 = 1
            and as_of_date in (SELECT(UNNEST(ARRAY{as_of_dates}::timestamp[])))
            and model_id = {model_id}
        """,
        "hash": "fghij",
    },
    {
        "name": "empty",
        "query": """\
            select entity_id, as_of_date
            from {results_schema}.predictions
            where entity_id = -1
            and as_of_date in (SELECT(UNNEST(ARRAY{as_of_dates}::timestamp[])))
            and model_id = {model_id}
        """,
        "hash": "abcde",
    },
]


def setup_results_schema(db_engine, num_entities, labels):
    # set up data, randomly generated by the factories but conforming
    # generally to what we expect model_metadata and results schemas data
    # to look like
    model = ModelFactory()

    class ImmediateTestPredictionFactory(TestPredictionFactory):
        as_of_date = factory.LazyAttribute(
            lambda o: o.model_rel.train_end_time
        )
    class ImmediateTrainPredictionFactory(TrainPredictionFactory):
        as_of_date = factory.LazyAttribute(
            lambda o: o.model_rel.train_end_time
        )
    
    for entity in range(0, num_entities):
        ImmediateTestPredictionFactory(
            model_rel=model,
            entity_id=entity,
            label_value=labels[entity],
        )
        ImmediateTrainPredictionFactory(
            model_rel=model,
            entity_id=entity,
            label_value=labels[entity],
        )

    session.commit()
        
    scores = []
    labels = []
    for row in db_engine.execute(
        """\
        select label_value, score
        from test_results.predictions
        order by entity_id
        """
    ):
        labels.append(row[0])
        scores.append(float(row[1]))

    model_metadata = [
        row 
        for row in db_engine.execute(
            """\
            select model_id, train_matrix_uuid, train_end_time
            from model_metadata.models
            """
        )
    ][0]

    return scores, model_metadata[0], model_metadata[1], model_metadata[2]


def fake_matrix_from_db(db_engine, matrix_type, model_id, num_entities, as_of_date):
        fake_matrix_store = MockMatrixStore(
            matrix_type,
            model_id,
            num_entities,
            db_engine,
            init_labels=pandas.read_sql(
                "select entity_id, as_of_date, label_value from train_results.predictions",
                db_engine
            ).set_index(["entity_id", "as_of_date"]).label_value,
            init_as_of_dates=[as_of_date]
        )
        return fake_matrix_store


def test_all_same_labels():
    for label_value in [0, 1]:
        with testing.postgresql.Postgresql() as postgresql:
            db_engine = create_engine(postgresql.url())
            ensure_db(db_engine)
            init_engine(db_engine)

            num_entities = 5
            labels = [label_value] * num_entities

            scores, model_id, matrix_uuid, as_of_date = setup_results_schema(
                db_engine,
                num_entities,
                labels,
            )

            # We should be able to calculate accuracy even if all of the labels
            # are the same, but ROC_AUC requires some positive and some
            # negative labels, so we should get one warning and one NULL value
            # for this config
            training_metric_groups = [{"metrics": ["accuracy", "roc_auc"]}]
            
            # Acquire fake data and objects to be used in the tests
            model_evaluator = ModelEvaluator(
                {},
                training_metric_groups,
                db_engine,
            )
            fake_matrix_store = fake_matrix_from_db(db_engine, "train", matrix_uuid, num_entities, as_of_date)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                model_evaluator.evaluate(scores, fake_matrix_store, model_id)
                assert len(w) == 1
                assert issubclass(w[-1].category, RuntimeWarning)
                assert "NULL" in str(w[-1].message)
                
                for row in db_engine.execute(
                    f"""select metric, value
                    from train_results.evaluations
                    where model_id = %s and
                    evaluation_start_time = %s
                    order by 1""",
                    (
                        model_id,
                        fake_matrix_store.as_of_dates[0]
                    ),
                ):
                    if row[0] == "accuracy":
                        assert row[1] is not None
                    else:
                        assert row[1] is None


def test_subset_labels_and_predictions():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        init_engine(db_engine)

        num_entities = 5
        labels = [0, 1, 0, 1, 0]

        scores, model_id, matrix_uuid, as_of_date = setup_results_schema(
            db_engine,
            num_entities,
            labels,
        )

        model_evaluator = ModelEvaluator({}, {}, db_engine)
        fake_matrix_store = fake_matrix_from_db(db_engine, "test", matrix_uuid, num_entities, as_of_date)

        for subset in SUBSETS:
            if subset["name"] == "evens":
                expected_result = 3
            elif subset["name"] == "odds":
                expected_result = 2
            elif subset["name"] == "empty":
                expected_result = 0

            subset_labels, subset_predictions = model_evaluator._subset_labels_and_predictions(
                fake_matrix_store,
                scores,
                subset["query"],
                model_id,
            )

            assert len(subset_labels) == expected_result
            assert len(subset_predictions) == expected_result


def test_evaluating_early_warning():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        init_engine(db_engine)

        num_entities = 10
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

        scores, model_id, matrix_uuid, as_of_date = setup_results_schema(
            db_engine,
            num_entities,
            labels,
        )

        # Set up testing configuration parameters
        testing_metric_groups = [
            {
                "metrics": [
                    "precision@",
                    "recall@",
                    "true positives@",
                    "true negatives@",
                    "false positives@",
                    "false negatives@",
                ],
                "thresholds": {"percentiles": [5.0, 10.0], "top_n": [5, 10]},
            },
            {
                "metrics": [
                    "f1",
                    "mediocre",
                    "accuracy",
                    "roc_auc",
                    "average precision score",
                ]
            },
            {"metrics": ["fbeta@"], "parameters": [{"beta": 0.75}, {"beta": 1.25}]},
        ]

        training_metric_groups = [{"metrics": ["accuracy", "roc_auc"]}]

        custom_metrics = {"mediocre": always_half}

        # Acquire fake data and objects to be used in the tests
        model_evaluator = ModelEvaluator(
            testing_metric_groups,
            training_metric_groups,
            db_engine,
            custom_metrics=custom_metrics,
        )

        
        fake_test_matrix_store = fake_matrix_from_db(db_engine, "test", matrix_uuid, num_entities, as_of_date)
        fake_train_matrix_store = fake_matrix_from_db(db_engine, "train", matrix_uuid, num_entities, as_of_date)


        # Run tests for overall and subset evaluations
        for subset in [None] + SUBSETS:
            if subset is None:
                where_hash = ""
            else:
                SubsetFactory(subset_hash=subset["hash"])
                session.commit()
                where_hash = f"and subset_hash = '{subset['hash']}'"

            # Evaluate the testing metrics and test for all of them.
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model_evaluator.evaluate(
                    scores,
                    fake_test_matrix_store,
                    model_id,
                    subset,
                )
                records = [
                    row[0]
                    for row in db_engine.execute(
                        f"""\
                        select distinct(metric || parameter)
                        from test_results.evaluations
                        where model_id = %s and
                        evaluation_start_time = %s
                        {where_hash}
                        order by 1
                        """,
                        (
                            model_id,
                            fake_test_matrix_store.as_of_dates[0]
                        ),
                    )
                ]
                assert records == [
                    "accuracy",
                    "average precision score",
                    "f1",
                    "false negatives@10.0_pct",
                    "false negatives@10_abs",
                    "false negatives@5.0_pct",
                    "false negatives@5_abs",
                    "false positives@10.0_pct",
                    "false positives@10_abs",
                    "false positives@5.0_pct",
                    "false positives@5_abs",
                    "fbeta@0.75_beta",
                    "fbeta@1.25_beta",
                    "mediocre",
                    "precision@10.0_pct",
                    "precision@10_abs",
                    "precision@5.0_pct",
                    "precision@5_abs",
                    "recall@10.0_pct",
                    "recall@10_abs",
                    "recall@5.0_pct",
                    "recall@5_abs",
                    "roc_auc",
                    "true negatives@10.0_pct",
                    "true negatives@10_abs",
                    "true negatives@5.0_pct",
                    "true negatives@5_abs",
                    "true positives@10.0_pct",
                    "true positives@10_abs",
                    "true positives@5.0_pct",
                    "true positives@5_abs",
                ]
                if subset is not None and subset["name"] == "empty":
                    assert issubclass(w[-1].category, RuntimeWarning)
                    assert "NULL" in str(w[-1].message)

            # Evaluate the training metrics and test
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                model_evaluator.evaluate(
                    scores,
                    fake_train_matrix_store,
                    model_id,
                    subset,
                )
                
                records = [
                    row[0]
                    for row in db_engine.execute(
                        f"""select distinct(metric || parameter)
                        from train_results.evaluations
                        where model_id = %s and
                        evaluation_start_time = %s
                        {where_hash}
                        order by 1""",
                        (
                            model_id,
                            fake_train_matrix_store.as_of_dates[0]
                        ),
                    )
                ]
                assert records == ["accuracy", "roc_auc"]
                if subset is not None and subset["name"] == "empty":
                    assert issubclass(w[-1].category, RuntimeWarning)
                    assert "NULL" in str(w[-1].message)


def test_model_scoring_inspections():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)
        testing_metric_groups = [
            {
                "metrics": ["precision@", "recall@", "fpr@"],
                "thresholds": {"percentiles": [50.0], "top_n": [3]},
            },
            {
                # ensure we test a non-thresholded metric as well
                "metrics": ["accuracy"]
            },
        ]
        training_metric_groups = [
            {"metrics": ["accuracy"], "thresholds": {"percentiles": [50.0]}}
        ]

        model_evaluator = ModelEvaluator(
            testing_metric_groups, training_metric_groups, db_engine
        )

        testing_labels = numpy.array([True, False, numpy.nan, True, False])
        testing_prediction_probas = numpy.array([0.56, 0.4, 0.55, 0.5, 0.3])

        training_labels = numpy.array(
            [False, False, True, True, True, False, True, True]
        )
        training_prediction_probas = numpy.array(
            [0.6, 0.4, 0.55, 0.70, 0.3, 0.2, 0.8, 0.6]
        )

        fake_train_matrix_store = MockMatrixStore(
            "train", "efgh", 5, db_engine, training_labels
        )
        fake_test_matrix_store = MockMatrixStore(
            "test", "1234", 5, db_engine, testing_labels
        )

        trained_model, model_id = fake_trained_model(db_engine)

        # Evaluate testing matrix and test the results
        model_evaluator.evaluate(
            testing_prediction_probas, fake_test_matrix_store, model_id
        )
        for record in db_engine.execute(
            """select * from test_results.evaluations
            where model_id = %s and evaluation_start_time = %s
            order by 1""",
            (model_id, fake_test_matrix_store.as_of_dates[0]),
        ):
            assert record["num_labeled_examples"] == 4
            assert record["num_positive_labels"] == 2
            if record["parameter"] == "":
                assert record["num_labeled_above_threshold"] == 4
            elif "pct" in record["parameter"]:
                assert record["num_labeled_above_threshold"] == 1
            else:
                assert record["num_labeled_above_threshold"] == 2

        # Evaluate the training matrix and test the results
        model_evaluator.evaluate(
            training_prediction_probas, fake_train_matrix_store, model_id
        )
        for record in db_engine.execute(
            """select * from train_results.evaluations
            where model_id = %s and evaluation_start_time = %s
            order by 1""",
            (model_id, fake_train_matrix_store.as_of_dates[0]),
        ):
            assert record["num_labeled_examples"] == 8
            assert record["num_positive_labels"] == 5
            assert record["value"] == 0.625


def test_ModelEvaluator_needs_evaluation(db_engine):
    ensure_db(db_engine)
    init_engine(db_engine)
    # TEST SETUP:

    # create two models: one that has zero evaluations,
    # one that has an evaluation for precision@100_abs
    # both overall and for each subset
    model_with_evaluations = ModelFactory()
    model_without_evaluations = ModelFactory()

    eval_time = datetime.datetime(2016, 1, 1)
    as_of_date_frequency = "3d"
    for subset_hash in [''] + [subset['hash'] for subset in SUBSETS]:
        EvaluationFactory(
            model_rel=model_with_evaluations,
            evaluation_start_time=eval_time,
            evaluation_end_time=eval_time,
            as_of_date_frequency=as_of_date_frequency,
            metric="precision@",
            parameter="100_abs",
            subset_hash=subset_hash,
        )
    session.commit()

    # make a test matrix to pass in
    metadata_overrides = {
        'as_of_date_frequency': as_of_date_frequency,
        'as_of_times': [eval_time],
    }
    test_matrix_store = MockMatrixStore(
        "test", "1234", 5, db_engine, metadata_overrides=metadata_overrides
    )
    train_matrix_store = MockMatrixStore(
        "train", "2345", 5, db_engine, metadata_overrides=metadata_overrides
    )

    # the evaluated model has test evaluations for precision, but not recall,
    # so this needs evaluations
    for subset in [None] + SUBSETS:
        if not subset:
            subset_hash = ''
        else:
            subset_hash = subset['hash']
        
        assert ModelEvaluator(
            testing_metric_groups=[{
                "metrics": ["precision@", "recall@"],
                "thresholds": {"top_n": [100]},
            }],
            training_metric_groups=[],
            db_engine=db_engine,
        ).needs_evaluations(
            matrix_store=test_matrix_store,
            model_id=model_with_evaluations.model_id,
            subset_hash=subset_hash,
        )

    # the evaluated model has test evaluations for precision,
    # so this should not need evaluations
    for subset in [None] + SUBSETS:
        if not subset:
            subset_hash = ''
        else:
            subset_hash = subset['hash']

        assert not ModelEvaluator(
            testing_metric_groups=[{
                "metrics": ["precision@"],
                "thresholds": {"top_n": [100]},
            }],
            training_metric_groups=[],
            db_engine=db_engine,
        ).needs_evaluations(
            matrix_store=test_matrix_store,
            model_id=model_with_evaluations.model_id,
            subset_hash=subset_hash,
        )

    # the non-evaluated model has no evaluations,
    # so this should need evaluations
    for subset in [None] + SUBSETS:
        if not subset:
            subset_hash = ''
        else:
            subset_hash = subset['hash']
        
        assert ModelEvaluator(
            testing_metric_groups=[{
                "metrics": ["precision@"],
                "thresholds": {"top_n": [100]},
            }],
            training_metric_groups=[],
            db_engine=db_engine,
        ).needs_evaluations(
            matrix_store=test_matrix_store,
            model_id=model_without_evaluations.model_id,
            subset_hash=subset_hash,
        )

    # the evaluated model has no *train* evaluations,
    # so the train matrix should need evaluations
    for subset in [None] + SUBSETS:
        if not subset:
            subset_hash = ''
        else:
            subset_hash = subset['hash']
        
        assert ModelEvaluator(
            testing_metric_groups=[{
                "metrics": ["precision@"],
                "thresholds": {"top_n": [100]},
            }],
            training_metric_groups=[{
                "metrics": ["precision@"],
                "thresholds": {"top_n": [100]},
            }],
            db_engine=db_engine
        ).needs_evaluations(
            matrix_store=train_matrix_store,
            model_id=model_with_evaluations.model_id,
            subset_hash=subset_hash,
        )
    session.close()
    session.remove()


def test_generate_binary_at_x():
    input_list = [0.9, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6]

    # bug can arise when the same value spans both sides of threshold
    assert generate_binary_at_x(input_list, 50, "percentile") == [
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    ]

    assert generate_binary_at_x(input_list, 2) == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
