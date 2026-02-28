from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, evaluate_model
from airflow import configuration as conf

# Enable pickle support so XCom can pass serialized objects between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# ── Default Arguments ─────────────────────────────────────────────────────────
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ── DAG Definition ────────────────────────────────────────────────────────────
dag = DAG(
    'sales_churn_prediction',
    default_args=default_args,
    description='Logistic Regression pipeline to predict customer churn from sales data',
    schedule_interval=None,   # Manual trigger; swap for '@daily' if needed
    catchup=False,
)

# ── Task 1: Load Data ─────────────────────────────────────────────────────────
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# ── Task 2: Preprocess Data ───────────────────────────────────────────────────
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],   # receives serialized DataFrame
    dag=dag,
)

# ── Task 3: Train & Save Model ────────────────────────────────────────────────
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, '/opt/airflow/working_data/churn_model.pkl'],
    provide_context=True,
    dag=dag,
)

# ── Task 4: Evaluate Model ────────────────────────────────────────────────────
evaluate_model_task = PythonOperator(
    task_id='evaluate_model_task',
    python_callable=evaluate_model,
    op_args=[
        '/opt/airflow/working_data/churn_model.pkl',
        build_save_model_task.output
    ],
    dag=dag,
)

# ── Task Dependencies ─────────────────────────────────────────────────────────
load_data_task >> data_preprocessing_task >> build_save_model_task >> evaluate_model_task

if __name__ == "__main__":
    dag.cli()