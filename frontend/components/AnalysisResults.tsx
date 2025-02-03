import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  List,
  ListItem,
  ListItemText,
  Divider,
  Button,
  Chip,
} from '@mui/material';

interface AnalysisResultsProps {
  results: {
    property_info: any;
    building_history: any[];
    zoning_info: any;
    development_potential: {
      basement_rental: any;
      attic_conversion: any;
      property_division: any;
      recommendations: string[];
      estimated_costs: Record<string, number>;
      estimated_value_increase: number;
    };
    enova_support: {
      eligible_measures: string[];
      potential_support: number;
      energy_savings: number;
      requirements: string[];
    };
  };
}

export const AnalysisResults: React.FC<AnalysisResultsProps> = ({ results }) => {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('nb-NO', {
      style: 'currency',
      currency: 'NOK',
    }).format(amount);
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Grid container spacing={3}>
        {/* Eiendomsinformasjon */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Eiendomsinformasjon
            </Typography>
            <List>
              <ListItem>
                <ListItemText
                  primary="Adresse"
                  secondary={results.property_info.address}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Kommune"
                  secondary={results.property_info.municipality}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="GNR/BNR"
                  secondary={`${results.property_info.gnr}/${results.property_info.bnr}`}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Areal"
                  secondary={`${results.property_info.area} m²`}
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        {/* Reguleringsinfo */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Regulering og Bestemmelser
            </Typography>
            <List>
              <ListItem>
                <ListItemText
                  primary="Plan ID"
                  secondary={results.zoning_info.plan_id}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Utnyttelsesgrad"
                  secondary={`${results.zoning_info.coverage_rate}% BYA`}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Maksimal høyde"
                  secondary={`${results.zoning_info.max_height} meter`}
                />
              </ListItem>
            </List>
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Spesielle bestemmelser:
              </Typography>
              {results.zoning_info.special_regulations.map((reg: string, index: number) => (
                <Chip
                  key={index}
                  label={reg}
                  sx={{ m: 0.5 }}
                  size="small"
                />
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* Utviklingspotensial */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Utviklingspotensial
            </Typography>
            <Grid container spacing={3}>
              {results.development_potential.basement_rental && (
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle1" gutterBottom>
                    Kjeller Utleie
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemText
                        primary="Mulig leieinntekt"
                        secondary={formatCurrency(
                          results.development_potential.basement_rental.potential_income
                        )}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Estimerte kostnader"
                        secondary={formatCurrency(
                          results.development_potential.basement_rental.estimated_cost
                        )}
                      />
                    </ListItem>
                  </List>
                </Grid>
              )}

              {results.development_potential.attic_conversion && (
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle1" gutterBottom>
                    Loft Konvertering
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemText
                        primary="Potensielt boareal"
                        secondary={`${results.development_potential.attic_conversion.potential_area} m²`}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Estimerte kostnader"
                        secondary={formatCurrency(
                          results.development_potential.attic_conversion.estimated_cost
                        )}
                      />
                    </ListItem>
                  </List>
                </Grid>
              )}

              {results.development_potential.property_division && (
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle1" gutterBottom>
                    Tomtedeling
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemText
                        primary="Mulig tomt"
                        secondary={`${results.development_potential.property_division.potential_area} m²`}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Estimert verdi"
                        secondary={formatCurrency(
                          results.development_potential.property_division.estimated_value
                        )}
                      />
                    </ListItem>
                  </List>
                </Grid>
              )}
            </Grid>

            <Divider sx={{ my: 3 }} />

            <Typography variant="h6" gutterBottom>
              Anbefalinger
            </Typography>
            <List>
              {results.development_potential.recommendations.map((rec, index) => (
                <ListItem key={index}>
                  <ListItemText primary={rec} />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        {/* Enova Støtte */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Enova Støttemuligheter
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Tilgjengelige tiltak:
                </Typography>
                <List>
                  {results.enova_support.eligible_measures.map((measure, index) => (
                    <ListItem key={index}>
                      <ListItemText primary={measure} />
                    </ListItem>
                  ))}
                </List>
              </Grid>
              <Grid item xs={12} md={6}>
                <List>
                  <ListItem>
                    <ListItemText
                      primary="Potensiell støtte"
                      secondary={formatCurrency(results.enova_support.potential_support)}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Energibesparelse"
                      secondary={`${results.enova_support.energy_savings} kWh/år`}
                    />
                  </ListItem>
                </List>
              </Grid>
            </Grid>

            <Box sx={{ mt: 3 }}>
              <Button
                variant="contained"
                color="primary"
                onClick={() => {
                  // Implementer nedlasting av Enova-søknad
                }}
              >
                Last ned Enova-søknad
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Handlingsknapper */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mt: 2 }}>
            <Button
              variant="contained"
              onClick={() => {
                // Implementer nedlasting av komplett rapport
              }}
            >
              Last ned komplett rapport
            </Button>
            <Button
              variant="contained"
              onClick={() => {
                // Implementer nedlasting av byggesaksdokumenter
              }}
            >
              Last ned byggesaksdokumenter
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};