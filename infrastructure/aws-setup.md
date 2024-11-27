# AWS Setup for Eiendomsmuligheter

## 1. AWS Konto Oppsett
1. Gå til aws.amazon.com og opprett en ny AWS-konto
2. Velg "Business"-konto
3. Aktiver AWS Organizations for beste praksis sikkerhet
4. Sett opp MFA (Multi-Factor Authentication) for root-kontoen
5. Opprett IAM-brukere for administratorer

## 2. Regional Konfigurasjon
### Primær Region: Stockholm (eu-north-1)
- Nærmeste region til Norge
- Lavest latens for norske brukere
- Oppfyller EUs databeskyttelseskrav

### Backup Region: Frankfurt (eu-central-1)
- Sekundær region for disaster recovery
- Høy tilgjengelighet og redundans
- Oppfyller også EUs databeskyttelseskrav

## 3. Kostnadsestimat (månedlig)
### Basis Infrastruktur
- EC2 Instances (t3.xlarge x 2): ~$150
- RDS for MongoDB: ~$200
- Elastic Load Balancer: ~$30
- Route 53: ~$1
- CloudFront: ~$50
- S3 Storage: ~$23
- Lambda Functions: ~$20
- Elastic IP: ~$4

### AI/ML Infrastruktur
- SageMaker Instances: ~$200
- GPU Instances (p3.2xlarge) ved behov: ~$3/time

### Backup og Sikkerhet
- AWS Backup: ~$30
- AWS WAF: ~$10
- AWS Shield Standard: Inkludert
- Certificate Manager: Gratis

Estimert Total: ~$700-1000/måned
Note: Kostnader vil variere basert på faktisk bruk og trafikk

## 4. Sikkerhetstiltak
1. Implementer AWS WAF for webbeskyttelse
2. Aktiver AWS Shield for DDoS-beskyttelse
3. Bruk AWS Security Hub for sikkerhetsoversikt
4. Implementer AWS Config for compliance-sporing
5. Sett opp CloudTrail for logging

## 5. Compliance
1. Implementer tagging-strategi for kostnadskontroll
2. Sett opp automatisk backup
3. Konfigurer data retention policies
4. Implementer kryptering i hvile og transit
5. Sett opp logging og monitoring

## 6. Første Steg for Implementering
1. Opprett AWS-konto
2. Sikre root-kontoen med MFA
3. Opprett administrative IAM-brukere
4. Sett opp VPC i Stockholm-regionen
5. Implementer basis sikkerhetstiltak
6. Start med basis infrastruktur