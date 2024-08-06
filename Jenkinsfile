pipeline {
    agent {
        label 'docker-agent-python'
    }
    triggers {
        pollSCM('*/5 * * * *')
    }
    environment {
        WORKING_DIRECTORY = "${env.WORKSPACE}"
    }
    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out the code...'
                checkout scm
            }
        }
        stage('Prepare Virtual Environment') {
            steps {
                echo 'Preparing virtual environment...'
                sh """
                cd ${WORKING_DIRECTORY}
                python -m venv venv
                source venv/bin/activate
                pip install --upgrade pip
                """
            }
        }
        stage('Install Tox') {
            steps {
                echo 'Installing Tox...'
                sh """
                cd ${WORKING_DIRECTORY}/serving_api
                source ${WORKING_DIRECTORY}/venv/bin/activate
                pip install tox
                """
            }
        }
        stage('Test Application API') {
            steps {
                echo 'Running application api tests...'
                sh """
                cd ${WORKING_DIRECTORY}/serving_api
                source ${WORKING_DIRECTORY}/venv/bin/activate
                tox
                """
            }
        }
        stage('Deploy Application via Railway') {
            when {
                branch 'master'
                branch 'demo'
            }
            steps {
                echo 'Deploying application via Railway...'
                sh """
                npm i -g @railway/cli
                railway up --detach
                """
            }
        
        }
    }
    post {
        always {
            echo 'Cleaning up...'
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
