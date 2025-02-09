#!/bin/bash

# Add Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Create monitoring namespace
kubectl create namespace monitoring

# Install Prometheus
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  -f prometheus-values.yaml

# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  -f grafana-values.yaml

# Wait for deployments
kubectl rollout status deployment/prometheus-server -n monitoring
kubectl rollout status deployment/grafana -n monitoring

echo "Monitoring stack installed successfully!"
echo "Access Grafana at: https://monitoring.eiendomsmuligheter.no"