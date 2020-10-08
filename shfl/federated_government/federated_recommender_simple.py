import numpy as np

from shfl.federated_government.federated_recommender import FederatedRecommender


class SimpleFederatedRecommender(FederatedRecommender):
    """
    Class used to represent the central class FederatedRecommender.

    # Arguments:
       model_builder: Function that return a trainable model (see: [Model](../model))
       federated_data: Federated data to use.
       (see: [FederatedData](../private/federated_operation/#federateddata-class))
       aggregator: Federated aggregator function (see: [Federated Aggregator](../federated_aggregator))
       model_param_access: Policy to access model's parameters, by default non-protected
       (see: [DataAccessDefinition](../private/data/#dataaccessdefinition-class))

    # Properties:
        global_model: Return the global model.

    The training and test data are arrays such that the first column denotes the clientId to which that data belongs.
    Every model in a node must contain the clientId property.
    """

    def __init__(self, model_builder, federated_data):
        super().__init__(model_builder, model_builder, federated_data, aggregator=None, model_params_access=None)

    def deploy_central_model(self):
        for data_node in self._federated_data:
            data_node.set_model_params(self._model.get_model_params())

    def evaluate_global_model_recommender(self, data_test, label_test, evaluations):
        """
        Method that evaluates the performance of the global model using the evaluation of each client.

        # Arguments:
            test_data: test dataset. The first column corresponds to the clientId.
            test_label: corresponding labels to test dataset.
            evaluations: dictionary containing the evaluation of the clients.
        """
        num_test = 0
        squared_error = 0
        for client in np.unique(data_test[:, 0]):
            eva_tmp = evaluations[client]
            num_test += len(data_test[data_test[:, 0] == client])
            squared_error += eva_tmp ** 2 * len(data_test[data_test[:, 0] == client])
        rmse = np.sqrt(squared_error / num_test)
        print("RMSE in the test set: {:.2f}".format(rmse))

    def run_rounds(self, test_data, test_label):
        """
        # Arguments:
            test_data: Test data for evaluation between rounds
            test_label: Test label for evaluation between rounds
        """
        self.deploy_central_model()
        self.train_all_clients()
        self.evaluate_global_model(test_data, test_label)
