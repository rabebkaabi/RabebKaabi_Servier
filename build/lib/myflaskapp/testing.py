import unittest
from app import app

class TestApp(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()

    def tearDown(self):
        pass

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Molecule Prediction', response.data)

    def test_predict_route(self):
        data = {'smile': 'C'}
        response = self.app.post('/predict', json=data)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('prediction_model1', data)
        self.assertIn('prediction_model2', data)

    def test_model3_predictions(self):
        smiles = ['C', 'CC', 'CCC']  # Example SMILES strings
        for smile in smiles:
            data = {'smile': smile}
            response = self.app.post('/predict', json=data)
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertIn('prediction_model3', data)
            self.assertIn('P1', data['prediction_model3'])  # Adjust for the specific property name

if __name__ == '__main__':
    unittest.main()
