import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  List,
  ListItem,
  ListItemText,
  Divider,
  Button,
} from '@mui/material';

interface AnalysisResultsProps {
  results: {
    propertyInfo: {
      address: string;
      size: number;
      yearBuilt: number;
      propertyType: string;
    };
    regulations: {
      zoning: string;
      maxUtilization: number;
      heightLimit: number;
      currentUtilization: number;
    };
    developmentPotential: {
      options: Array<{
        type: string;
        description: string;
        feasibility: number;
        requirements: string[];
      }>;
    };
    energyAnalysis: {
      currentRating: string;
      potentialRating: string;
      recommendations: Array<{
        measure: string;
        cost: number;
        savings: number;
        enovaSupport: number;
      }>;
    };
  };
}

export const AnalysisResults: React.FC<AnalysisResultsProps> = ({ results }) => {
  const handleGenerateReport = () => {
    // Implementer rapport-generering
  };

  const handleDownloadDocuments = () => {
    // Implementer dokument-nedlasting
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        Analyserapport
      </Typography>

      <Grid container spacing={3}>
        {/* Eiendomsinformasjon */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Eiendomsinformasjon
              </Typography>
              <List>
                <ListItem>
                  <ListItemText
                    primary="Adresse"
                    secondary={results.propertyInfo.address}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Størrelse"
                    secondary={`${results.propertyInfo.size} m²`}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Byggeår"
                    secondary={results.propertyInfo.yearBuilt}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Type"
                    secondary={results.propertyInfo.propertyType}
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Regulering */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Regulering
              </Typography>
              <List>
                <ListItem>
                  <ListItemText
                    primary="Reguleringsformål"
                    secondary={results.regulations.zoning}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Maksimal utnyttelse"
                    secondary={`${results.regulations.maxUtilization}% BYA`}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Nåværende utnyttelse"
                    secondary={`${results.regulations.currentUtilization}% BYA`}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Høydebegrensning"
                    secondary={`${results.regulations.heightLimit} meter`}
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Utviklingspotensial */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Utviklingspotensial
              </Typography>
              {results.developmentPotential.options.map((option, index) => (
                <Box key={index} sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    {option.type} - {option.feasibility}% gjennomførbart
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {option.description}
                  </Typography>
                  <List dense>
                    {option.requirements.map((req, reqIndex) => (
                      <ListItem key={reqIndex}>
                        <ListItemText primary={`• ${req}`} />
                      </ListItem>
                    ))}
                  </List>
                  <Divider sx={{ mt: 1 }} />
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* Energianalyse */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Energianalyse
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle1">
                    Nåværende energimerking: {results.energyAnalysis.currentRating}
                  </Typography>
                  <Typography variant="subtitle1">
                    Potensial energimerking: {results.energyAnalysis.potentialRating}
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Anbefalte tiltak
                  </Typography>
                  <List>
                    {results.energyAnalysis.recommendations.map((rec, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={rec.measure}
                          secondary={
                            <>
                              Kostnad: {rec.cost} kr | Besparelse: {rec.savings} kr/år
                              <br />
                              Enova-støtte: {rec.enovaSupport} kr
                            </>
                          }
                        />
                      </ListItem>
                    ))}
                  </List>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Handlingsknapper */}
      <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleGenerateReport}
        >
          Generer fullstendig rapport
        </Button>
        <Button
          variant="outlined"
          color="primary"
          onClick={handleDownloadDocuments}
        >
          Last ned byggesøknadsdokumenter
        </Button>
      </Box>
    </Box>
  );
};