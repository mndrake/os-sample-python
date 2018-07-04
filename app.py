from flask import Flask
from flask_restplus import Api, fields, Resource

from score import score

app = Flask(__name__)
api = Api(app, version='1.0', default='survive_titanic', title='Titanic API', description='A simple Prediction API')
ns = api.namespace('survive_titanic', description='Predict Titanic Survival')

parser = api.parser()
parser.add_argument('Age', type=float, help='Age of Passenger', location='form')
parser.add_argument('Pclass', type=int, help='Passenger Class', choices=(1, 2, 3), location='form')
parser.add_argument('Fare', type=float, help='Ticket Fare', location='form')
parser.add_argument('Parch', type=int, help='Number of Parents/Children Aboard', location='form')
parser.add_argument('Sex', type=str, help='Sex of Passenger', choices=('female', 'male'), location='form')
parser.add_argument('SibSp', type=int, help='Number of Siblings/Spouses Aboard', location='form')

resource_fields = api.model('Resource', {'result': fields.Float})


@ns.route('/')
class TitanicApi(Resource):

   @api.doc(parser=parser)
   @api.marshal_with(resource_fields)
   def post(self):
     args = parser.parse_args()
     return self.get_result(args), 201

   def get_result(self, args):
        y_pred = score(args)
        return {'result': y_pred}

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
