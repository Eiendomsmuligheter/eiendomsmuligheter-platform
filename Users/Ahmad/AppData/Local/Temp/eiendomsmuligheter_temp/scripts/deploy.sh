#!/bin/bash

# Farger for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Starting deployment process...${NC}"

# Sjekk om AWS CLI er installert
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Sjekk om docker er installert
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install it first.${NC}"
    exit 1
fi

# Sett miljøvariabler
export AWS_DEFAULT_REGION=eu-north-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_FRONTEND="$ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/eiendomsmuligheter-frontend"
ECR_BACKEND="$ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/eiendomsmuligheter-backend"

echo -e "${YELLOW}Building and pushing Docker images...${NC}"

# Logg inn til ECR
aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com

# Bygg og push frontend
echo -e "${YELLOW}Building frontend...${NC}"
docker build -t $ECR_FRONTEND:latest -f frontend/Dockerfile.prod frontend/
docker push $ECR_FRONTEND:latest

# Bygg og push backend
echo -e "${YELLOW}Building backend...${NC}"
docker build -t $ECR_BACKEND:latest -f backend/Dockerfile.prod backend/
docker push $ECR_BACKEND:latest

echo -e "${YELLOW}Updating ECS services...${NC}"

# Oppdater ECS services
aws ecs update-service --cluster eiendomsmuligheter --service frontend-service --force-new-deployment
aws ecs update-service --cluster eiendomsmuligheter --service backend-service --force-new-deployment

echo -e "${YELLOW}Waiting for services to stabilize...${NC}"

# Vent på at tjenestene skal bli stabile
aws ecs wait services-stable --cluster eiendomsmuligheter --services frontend-service backend-service

# Sjekk helsestatus for tjenestene
check_service_health() {
    local service_name=$1
    local health_status=$(aws ecs describe-services --cluster eiendomsmuligheter --services $service_name \
        --query 'services[0].deployments[0].rolloutState' --output text)
    
    if [ "$health_status" == "COMPLETED" ]; then
        echo -e "${GREEN}Service $service_name is healthy${NC}"
        return 0
    else
        echo -e "${RED}Service $service_name deployment failed or is incomplete${NC}"
        return 1
    fi
}

# Sjekk helsestatus for begge tjenester
check_service_health "frontend-service"
check_service_health "backend-service"

echo -e "${GREEN}Deployment completed successfully!${NC}"

# Vis status for tjenestene
echo -e "${YELLOW}Current service status:${NC}"
aws ecs describe-services --cluster eiendomsmuligheter --services frontend-service backend-service \
    --query 'services[*].{Name:serviceName,Status:status,DesiredCount:desiredCount,RunningCount:runningCount}'