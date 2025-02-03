import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import PropertyAnalyzer from '../components/PropertyAnalyzer/PropertyAnalyzer';
import { PropertyAnalyzerProvider } from '../contexts/PropertyAnalyzerContext';

// Mock the property service
jest.mock('../services/propertyService', () => ({
  analyzeProperty: jest.fn().mockResolvedValue({
    property_info: {
      address: 'Test Address',
      municipality_code: '3005'
    },
    building_analysis: {
      building_type: 'enebolig',
      stories: 2,
      has_basement: true
    }
  })
}));

describe('PropertyAnalyzer Component', () => {
  beforeEach(() => {
    render(
      <PropertyAnalyzerProvider>
        <PropertyAnalyzer />
      </PropertyAnalyzerProvider>
    );
  });

  test('renders all analysis options', () => {
    expect(screen.getByText(/søk med adresse/i)).toBeInTheDocument();
    expect(screen.getByText(/last opp bilder/i)).toBeInTheDocument();
    expect(screen.getByText(/angi finn\.no lenke/i)).toBeInTheDocument();
  });

  test('handles address input', async () => {
    // Find and click address option
    const addressButton = screen.getByText(/søk med adresse/i);
    fireEvent.click(addressButton);

    // Find and fill address input
    const addressInput = screen.getByPlaceholderText(/skriv inn adresse/i);
    await userEvent.type(addressInput, 'Testveien 1');

    // Submit form
    const submitButton = screen.getByText(/start analyse/i);
    fireEvent.click(submitButton);

    // Wait for results
    await waitFor(() => {
      expect(screen.getByText(/analyserer eiendom/i)).toBeInTheDocument();
    });
  });

  test('handles file upload', async () => {
    // Find and click file upload option
    const fileButton = screen.getByText(/last opp bilder/i);
    fireEvent.click(fileButton);

    // Create test file
    const file = new File(['dummy content'], 'house.jpg', { type: 'image/jpeg' });
    const fileInput = screen.getByLabelText(/last opp bilder/i);

    // Upload file
    await userEvent.upload(fileInput, file);

    // Check if file is accepted
    expect(screen.getByText(/house\.jpg/i)).toBeInTheDocument();
  });

  test('displays analysis results', async () => {
    // Start with address analysis
    const addressButton = screen.getByText(/søk med adresse/i);
    fireEvent.click(addressButton);

    const addressInput = screen.getByPlaceholderText(/skriv inn adresse/i);
    await userEvent.type(addressInput, 'Testveien 1');

    const submitButton = screen.getByText(/start analyse/i);
    fireEvent.click(submitButton);

    // Wait for and verify results
    await waitFor(() => {
      expect(screen.getByText(/testveien 1/i)).toBeInTheDocument();
      expect(screen.getByText(/enebolig/i)).toBeInTheDocument();
      expect(screen.getByText(/2 etasjer/i)).toBeInTheDocument();
    });
  });

  test('handles errors gracefully', async () => {
    // Mock a service error
    jest.spyOn(console, 'error').mockImplementation(() => {});
    jest.mock('../services/propertyService', () => ({
      analyzeProperty: jest.fn().mockRejectedValue(new Error('Analysis failed'))
    }));

    // Trigger analysis
    const addressButton = screen.getByText(/søk med adresse/i);
    fireEvent.click(addressButton);

    const addressInput = screen.getByPlaceholderText(/skriv inn adresse/i);
    await userEvent.type(addressInput, 'Testveien 1');

    const submitButton = screen.getByText(/start analyse/i);
    fireEvent.click(submitButton);

    // Check for error message
    await waitFor(() => {
      expect(screen.getByText(/noe gikk galt/i)).toBeInTheDocument();
    });
  });

  test('validates input before submission', async () => {
    // Click address option
    const addressButton = screen.getByText(/søk med adresse/i);
    fireEvent.click(addressButton);

    // Try to submit without address
    const submitButton = screen.getByText(/start analyse/i);
    fireEvent.click(submitButton);

    // Check for validation message
    expect(screen.getByText(/adresse er påkrevd/i)).toBeInTheDocument();
  });

  test('supports multiple file uploads', async () => {
    // Find and click file upload option
    const fileButton = screen.getByText(/last opp bilder/i);
    fireEvent.click(fileButton);

    // Create test files
    const file1 = new File(['dummy content 1'], 'front.jpg', { type: 'image/jpeg' });
    const file2 = new File(['dummy content 2'], 'back.jpg', { type: 'image/jpeg' });
    const fileInput = screen.getByLabelText(/last opp bilder/i);

    // Upload multiple files
    await userEvent.upload(fileInput, [file1, file2]);

    // Check if both files are accepted
    expect(screen.getByText(/front\.jpg/i)).toBeInTheDocument();
    expect(screen.getByText(/back\.jpg/i)).toBeInTheDocument();
  });

  test('allows file removal before analysis', async () => {
    // Upload a file
    const fileButton = screen.getByText(/last opp bilder/i);
    fireEvent.click(fileButton);

    const file = new File(['dummy content'], 'house.jpg', { type: 'image/jpeg' });
    const fileInput = screen.getByLabelText(/last opp bilder/i);
    await userEvent.upload(fileInput, file);

    // Find and click remove button
    const removeButton = screen.getByLabelText(/fjern fil/i);
    fireEvent.click(removeButton);

    // Verify file is removed
    expect(screen.queryByText(/house\.jpg/i)).not.toBeInTheDocument();
  });

  test('shows progress during analysis', async () => {
    // Start address analysis
    const addressButton = screen.getByText(/søk med adresse/i);
    fireEvent.click(addressButton);

    const addressInput = screen.getByPlaceholderText(/skriv inn adresse/i);
    await userEvent.type(addressInput, 'Testveien 1');

    const submitButton = screen.getByText(/start analyse/i);
    fireEvent.click(submitButton);

    // Check for progress indicator
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    expect(screen.getByText(/analyserer/i)).toBeInTheDocument();

    // Wait for completion
    await waitFor(() => {
      expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    });
  });
});