provider "aws" {
  region = "eu-north-1"  # Stockholm region
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "eiendomsmuligheter-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["eu-north-1a", "eu-north-1b", "eu-north-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = false  # For high availability
  
  tags = {
    Environment = "production"
    Project     = "eiendomsmuligheter"
  }
}

# Security Groups
resource "aws_security_group" "web" {
  name        = "web"
  description = "Web tier security group"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Application Load Balancer
module "alb" {
  source  = "terraform-aws-modules/alb/aws"
  version = "~> 8.0"

  name = "eiendomsmuligheter-alb"

  load_balancer_type = "application"

  vpc_id          = module.vpc.vpc_id
  subnets         = module.vpc.public_subnets
  security_groups = [aws_security_group.web.id]

  target_groups = [
    {
      name             = "app-tg"
      backend_protocol = "HTTP"
      backend_port     = 80
      target_type      = "instance"
    }
  ]
}

# Auto Scaling Group
module "asg" {
  source  = "terraform-aws-modules/autoscaling/aws"
  version = "~> 6.0"

  name = "eiendomsmuligheter-asg"

  vpc_zone_identifier = module.vpc.private_subnets
  target_group_arns  = module.alb.target_group_arns
  health_check_type  = "ELB"
  min_size          = 2
  max_size          = 10
  desired_capacity  = 2

  launch_template_name        = "eiendomsmuligheter-lt"
  launch_template_description = "Launch template for Eiendomsmuligheter"
  update_default_version      = true

  image_id          = "ami-0989fb15ce71ba39e"  # Amazon Linux 2 AMI ID for eu-north-1
  instance_type     = "t3.xlarge"

  tags = {
    Environment = "production"
    Project     = "eiendomsmuligheter"
  }
}

# Route53 DNS
resource "aws_route53_zone" "main" {
  name = "eiendomsmuligheter.no"  # Endre til faktisk domene
}

resource "aws_route53_record" "www" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.eiendomsmuligheter.no"  # Endre til faktisk domene
  type    = "A"

  alias {
    name                   = module.alb.lb_dns_name
    zone_id               = module.alb.lb_zone_id
    evaluate_target_health = true
  }
}

# S3 Bucket for static assets and backups
resource "aws_s3_bucket" "static" {
  bucket = "eiendomsmuligheter-static"
  acl    = "private"

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  tags = {
    Environment = "production"
    Project     = "eiendomsmuligheter"
  }
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  default_root_object = "index.html"

  origin {
    domain_name = aws_s3_bucket.static.bucket_regional_domain_name
    origin_id   = "S3-${aws_s3_bucket.static.id}"

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.main.cloudfront_access_identity_path
    }
  }

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-${aws_s3_bucket.static.id}"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }

  restrictions {
    geo_restriction {
      restriction_type = "whitelist"
      locations        = ["NO"]  # Kun Norge
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true  # Oppdater med eget SSL-sertifikat senere
  }

  tags = {
    Environment = "production"
    Project     = "eiendomsmuligheter"
  }
}

# WAF Web ACL
resource "aws_wafv2_web_acl" "main" {
  name        = "eiendomsmuligheter-waf"
  description = "WAF for Eiendomsmuligheter"
  scope       = "REGIONAL"

  default_action {
    allow {}
  }

  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name               = "AWSManagedRulesCommonRuleSetMetric"
      sampled_requests_enabled  = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name               = "EiendomsmuligheterWAFMetric"
    sampled_requests_enabled  = true
  }
}

# Outputs
output "alb_dns_name" {
  description = "DNS name of ALB"
  value       = module.alb.lb_dns_name
}

output "cloudfront_domain_name" {
  description = "Domain name of CloudFront distribution"
  value       = aws_cloudfront_distribution.main.domain_name
}