import React, { useState } from 'react';
import { CardElement, useStripe, useElements } from '@stripe/stripe-react-js';
import styles from '../styles/PaymentModal.module.css';

const PaymentModal = ({ onClose, onSuccess, type, amount }) => {
    const stripe = useStripe();
    const elements = useElements();
    const [error, setError] = useState(null);
    const [processing, setProcessing] = useState(false);

    const handleSubmit = async (event) => {
        event.preventDefault();
        setProcessing(true);

        if (!stripe || !elements) {
            return;
        }

        const { error, paymentMethod } = await stripe.createPaymentMethod({
            type: 'card',
            card: elements.getElement(CardElement),
        });

        if (error) {
            setError(error.message);
            setProcessing(false);
            return;
        }

        try {
            let response;
            if (type === 'subscription') {
                response = await fetch('/api/payment/create-subscription', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    },
                    body: JSON.stringify({
                        paymentMethodId: paymentMethod.id,
                        priceId: amount // In this case, amount is the priceId
                    })
                });
            } else if (type === 'credits') {
                response = await fetch('/api/payment/purchase-credits', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    },
                    body: JSON.stringify({
                        paymentMethodId: paymentMethod.id,
                        amount
                    })
                });
            }

            const data = await response.json();
            
            if (response.ok) {
                onSuccess(data);
            } else {
                setError(data.message);
            }
        } catch (err) {
            setError('Det oppstod en feil ved behandling av betalingen');
        }

        setProcessing(false);
    };

    return (
        <div className={styles.modalOverlay}>
            <div className={styles.modalContent}>
                <button className={styles.closeButton} onClick={onClose}>×</button>
                <h2>{type === 'subscription' ? 'Velg abonnement' : 'Kjøp kreditter'}</h2>
                
                <form onSubmit={handleSubmit}>
                    <div className={styles.paymentElement}>
                        <CardElement options={{
                            style: {
                                base: {
                                    fontSize: '16px',
                                    color: '#424770',
                                    '::placeholder': {
                                        color: '#aab7c4',
                                    },
                                },
                                invalid: {
                                    color: '#9e2146',
                                },
                            },
                        }}/>
                    </div>

                    {error && <div className={styles.error}>{error}</div>}

                    <button
                        type="submit"
                        disabled={!stripe || processing}
                        className={styles.submitButton}
                    >
                        {processing ? 'Behandler...' : 'Betal'}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default PaymentModal;