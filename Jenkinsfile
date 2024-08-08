pipeline {
    agent {
        label 'docker-agent-python'
    }
    triggers {
        pollSCM('*/5 * * * *')
    }
    environment {
        WORKING_DIRECTORY = "${env.WORKSPACE}"
        IMAGE_TAG = "stable_diffusion:${env.BUILD_NUMBER}"

        AWS_REGION = 'eu-west-1'
        AWS_ACCOUNT_ID = 'aws-account-id'
        ECR_REPOSITORY = 'ecr-repo-name'
        AWS_CLUSTER_NAME = 'ecs-cluster-name'
        AWS_SERVICE_NAME = 'ecs-service-name'
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
        stage('Build Docker Image') {
            steps {
                echo 'Building Docker image...'
                sh """
                cd ${WORKING_DIRECTORY}
                docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .
                """
            }
        }
        stage('Push to ECR') {
            steps {
                echo 'Push Docker image to AWS ECR...'
                sh """
                aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
                aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} || aws ecr create-repository --repository-name ${ECR_REPOSITORY}
                docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}
                docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}
                """
            }
        }
        stage('Deploy to ECS') {
            when {
                branch 'main'
            }
            steps {
                echo 'Deploying Docker container to ECS...'
                sh """
                aws ecs update-service --cluster ${AWS_CLUSTER_NAME} --service ${AWS_SERVICE_NAME} --force-new-deployment --region ${AWS_REGION}
                """
            }
        }
        // stage('Deploy Application via Railway') {
        //     when {
        //         branch 'main'
        //     }
        //     steps {
        //         echo 'Deploying application via Railway...'
        //         sh """
        //         npm i -g @railway/cli
        //         railway up --detach
        //         """
        //     }
        // }
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
