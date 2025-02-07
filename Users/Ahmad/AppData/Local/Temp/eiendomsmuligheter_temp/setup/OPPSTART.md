# Oppstartsprosedyre for Eiendomsmuligheter.no

## 1. Domeneregistrering
1. Gå til norid.no eller en registrar som Domeneshop
2. Registrer domenet eiendomsmuligheter.no
3. Vent på godkjenning fra Norid
4. Noter ned DNS-servere for senere bruk med Route53

## 2. AWS-Konto Oppsett
1. Gå til aws.amazon.com/nb
2. Klikk på "Opprett en AWS-konto"
3. Velg "Professional" kontoplan
4. Fyll ut:
   - Firmanavn: Eiendomsmuligheter
   - Epost: [Din forretningsepost]
   - AWS Kontotype: Business
   - Faktureringsinformasjon

### Sikkerhetstiltak for AWS-konto
1. Aktiver MFA på root-kontoen
2. Opprett IAM-brukere for administrasjon
3. Sett opp AWS Organizations
4. Aktiver CloudTrail for logging
5. Sett opp AWS Config

## 3. Database Setup
1. Opprett MongoDB Atlas konto
2. Velg Dedicated Cluster i Stockholm-region
3. Sett opp VPC Peering med AWS
4. Konfigurer backup og sikkerhet

## 4. Betalingsintegrasjon
1. Opprett Stripe-konto for betalingshåndtering
2. Sett opp Vipps Bedrift som sekundær betalingsløsning
3. Integrer begge i systemet

## 5. Teknisk Infrastruktur

### Frontend (Next.js)
```bash
Domain: eiendomsmuligheter.no
CDN: AWS CloudFront
Hosting: AWS S3 + CloudFront
SSL: AWS Certificate Manager
```

### Backend (Node.js)
```bash
API Domain: api.eiendomsmuligheter.no
Server: AWS ECS med Fargate
Database: MongoDB Atlas
Caching: Redis Enterprise
```

### Sikkerhet
```bash
WAF: AWS WAF
DDoS Protection: AWS Shield
SSL/TLS: AWS Certificate Manager
Secrets Management: AWS Secrets Manager
```

## 6. Monitoring og Logging
1. AWS CloudWatch for metrics
2. ELK Stack for logging
3. New Relic for APM
4. StatusCake for ekstern monitoring

## 7. Initial Deployment Sjekkliste
- [ ] VPC og nettverksoppsett
- [ ] Security Groups
- [ ] Load Balancer konfigurasjon
- [ ] Auto Scaling Groups
- [ ] CI/CD pipeline
- [ ] Backup-rutiner
- [ ] Monitoring-oppsett
- [ ] SSL-sertifikater
- [ ] DNS-konfigurasjon

## 8. Testing og Kvalitetssikring
1. Enhetstesting
2. Integrasjonstesting
3. Ytelsestesting
4. Sikkerhetstesting
5. UAT (User Acceptance Testing)

## 9. Dokumentasjon
1. API-dokumentasjon
2. Systemarkitektur
3. Driftsprosedyrer
4. Brukerguider
5. Feilsøkingsguider

## 10. Support-system
1. Zendesk for kundesupport
2. Intercom for chat
3. FAQ-system
4. Dokumentasjonsportal

## 11. GDPR og Compliance
1. Personvernerklæring
2. Databehandleravtaler
3. Cookie-policy
4. Brukervilkår

## 12. Markedsføring og Lansering
1. SEO-optimalisering
2. Google Analytics setup
3. Marketingmateriell
4. Lanseringsplan

## Estimerte Månedlige Kostnader
- AWS Infrastruktur: ~$700-1000
- MongoDB Atlas: ~$200
- Redis Enterprise: ~$100
- New Relic: ~$100
- StatusCake: ~$50
- Zendesk: ~$100
- Intercom: ~$150
- SSL-sertifikater: Inkludert i AWS
- Totalt: ~$1400-1700/måned

## Kontaktinformasjon for Support
Support Email: support@eiendomsmuligheter.no
Teknisk Support: teknikk@eiendomsmuligheter.no
Salg: salg@eiendomsmuligheter.no