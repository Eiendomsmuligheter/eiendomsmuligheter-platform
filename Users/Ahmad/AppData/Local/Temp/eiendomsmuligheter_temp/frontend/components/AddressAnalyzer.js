import { useState } from 'react';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import toast from 'react-hot-toast';

export default function AddressAnalyzer() {
  const [address, setAddress] = useState('');
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const { user, token } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!address.trim()) {
      toast.error('Vennligst skriv inn en adresse');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/api/analyze`,
        { address },
        {
          headers: {
            Authorization: `Bearer ${token}`
          }
        }
      );

      setAnalysis(response.data.data);
      toast.success('Analyse fullført!');
    } catch (error) {
      toast.error(error.response?.data?.message || 'Noe gikk galt under analysen');
      if (error.response?.status === 401) {
        // Redirect to login if not authenticated
        router.push('/login');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="address" className="block text-sm font-medium text-gray-700">
            Eiendomsadresse
          </label>
          <div className="mt-1">
            <input
              type="text"
              id="address"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              placeholder="F.eks: Storgata 1, 0182 Oslo"
              className="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md"
              disabled={loading}
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${
            loading ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {loading ? 'Analyserer...' : 'Start analyse'}
        </button>
      </form>

      {analysis && (
        <div className="mt-8 bg-white shadow overflow-hidden sm:rounded-lg">
          <div className="px-4 py-5 sm:px-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900">
              Analyserapport for {analysis.address}
            </h3>
          </div>
          <div className="border-t border-gray-200">
            <dl>
              {analysis.possibilities.map((possibility, index) => (
                <div key={index} className={`${index % 2 === 0 ? 'bg-gray-50' : 'bg-white'} px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6`}>
                  <dt className="text-sm font-medium text-gray-500">
                    {possibility.type}
                  </dt>
                  <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                    <div>
                      <span className="font-bold">Gjennomførbarhet:</span> {possibility.feasibility}
                    </div>
                    <div>
                      <span className="font-bold">Estimert kostnad:</span> {possibility.estimatedCost} NOK
                    </div>
                    <div className="mt-2">
                      <span className="font-bold">Krav:</span>
                      <ul className="list-disc pl-5 mt-1">
                        {possibility.requirements.map((req, i) => (
                          <li key={i}>{req}</li>
                        ))}
                      </ul>
                    </div>
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      )}
    </div>
  );
}