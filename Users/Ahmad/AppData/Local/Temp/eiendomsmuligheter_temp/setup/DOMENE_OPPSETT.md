# Domeneoppsett for Eiendomsmuligheter.no

## 1. Domeneregistrering hos Domeneshop

1. Gå til https://www.domeneshop.no
2. Søk etter "eiendomsmuligheter.no"
3. Registrer domenet for minimum 1 år
4. Ved registrering, fyll ut:
   - Organisasjonsnummer
   - Firmanavn: Eiendomsmuligheter
   - Kontaktinformasjon

## 2. DNS-konfigurasjon i AWS Route53

### 2.1 Opprett Hosted Zone
```bash
# I AWS Console:
1. Gå til Route53
2. Opprett Hosted Zone for eiendomsmuligheter.no
3. Noter ned nameservers (NS records)
```

### 2.2 Hoveddomene DNS Records
```bash
# A Records
eiendomsmuligheter.no.     A   ALIAS $CloudFront_Distribution
www.eiendomsmuligheter.no.  A   ALIAS $CloudFront_Distribution

# API Subdomain
api.eiendomsmuligheter.no.  A   ALIAS $ALB_DNS_Name

# MX Records for Epost
eiendomsmuligheter.no.  MX  10 mx.domeneshop.no.
eiendomsmuligheter.no.  MX  20 mx2.domeneshop.no.

# TXT Records for SPF
eiendomsmuligheter.no.  TXT "v=spf1 include:_spf.domeneshop.no ~all"

# DMARC Record
_dmarc.eiendomsmuligheter.no.  TXT "v=DMARC1; p=reject; rua=mailto:teknikk@eiendomsmuligheter.no"
```

### 2.3 SSL-sertifikat Oppsett
1. Request sertifikat i AWS Certificate Manager
2. Validering via DNS
3. Implementer i CloudFront og ALB

## 3. Email Oppsett

### 3.1 Hovedepostadresser
```bash
support@eiendomsmuligheter.no
teknikk@eiendomsmuligheter.no
salg@eiendomsmuligheter.no
post@eiendomsmuligheter.no
```

### 3.2 Email Forwarding
Sett opp videresending til faktiske epostkontoer

## 4. DNS Propagering
- Vent 24-48 timer på full DNS-propagering
- Verifiser DNS-oppsettet med online verktøy
- Test alle epostadresser

## 5. Monitoring
- Sett opp DNS monitoring i StatusCake
- Konfigurer SSL sertifikat monitoring
- Sett opp uptime monitoring

## 6. Backup og Sikkerhet
- Aktiver DNSSEC
- Implementer CAA records
- Dokumenter DNS-konfigurasjon
- Sett opp backup av DNS records

## 7. Testing
```bash
# Kjør disse kommandoene for å verifisere oppsettet
dig eiendomsmuligheter.no
dig www.eiendomsmuligheter.no
dig api.eiendomsmuligheter.no
dig mx eiendomsmuligheter.no
dig txt eiendomsmuligheter.no
```

## 8. Vedlikehold
- Sett opp varsling for domene-utløp
- Dokumenter fornyelsesprosedyrer
- Lag disaster recovery plan for DNS

## Viktige Kontakter
- Domeneshop Support: 24 07 70 00
- AWS Support: Via AWS Console
- StatusCake Support: support@statuscake.com