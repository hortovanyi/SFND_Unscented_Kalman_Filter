#ifndef UKF_H
#define UKF_H

#include <Eigen/Dense>
#include "measurement_package.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::tuple;

class UKF {
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);


  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  // state covariance matrix
  MatrixXd P_;

  ///* Measurement dimensions
  int n_laser_z_;
  int n_radar_z_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  // Weights of sigma points
  VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;

  private:
  /*
   * Generate Sigma Points
   * @param n_x State Dimension
   * @param x State Vector
   * @param P State Coveriance Matrix
   */
  MatrixXd GenerateSigmaPoints(int n_x, const VectorXd& x, const MatrixXd& P);

  /*
   * Augment Sigma Points
   * @param n_x State dimension
   * @param n_aug Augmentation dimension
   * @param std_a Process noise standard deviation longitudinal acceleration in m/s^2
   * @param std_yawdd Process noise standard deviation yaw acceleration in rad/s^2
   * @param x State vector
   * @param P State covariance Matrix
   */
  MatrixXd AugmentedSigmaPoints(int n_x, int n_aug, double std_a,
                                double std_yawdd, const VectorXd& x,
                                const MatrixXd& P);

  /*
   * Predict Sigma Points
   * @param n_x State dimension
   * @param n_aug Augmentation dimension
   * @param delta_t Time between k and k+1 in s
   * @param Xsig_aug Augmented sigma points matrix
   */
  MatrixXd PredictSigmaPoints(int n_x, int n_aug, double delta_t,
                              const MatrixXd& Xsig_aug);

  /*
   * Predict Mean and Covariance
   * @param n_x State dimension
   * @param n_aug Augmentation dimension
   * @param Xsig_pred Predicted sigma points matrix
   */
  tuple<VectorXd, MatrixXd> PredictMeanAndCovariance(int n_x, int n_aug,
                                                     const MatrixXd& Xsig_pred);

  /*
   * Predict Radar Measurement
   * @param n_x State dimension
   * @param n_aug Augmentation dimension
   * @param n_z Measurement dimension, radar can measure r, phi, and r_dot
   * @param std_radr Radar measurement noise standard deviation radius in m
   * @param std_radphi Radar measurement noise standard deviation angle in rad
   * @param std_radrd Radar measurement noise standard deviation radius change in m/s
   * @param Xsig_pred Predicted sigma points matrix
   */
  tuple<VectorXd, MatrixXd> PredictRadarMeasurement(int n_x, int n_aug, int n_z,
                                                    double std_radr,
                                                    double std_radphi,
                                                    double std_radrd,
                                                    const MatrixXd& Xsig_pred);

  /*
   * Predict Laser Measurement
   * @param n_x State dimension
   * @param n_aug Augmentation dimension
   * @param n_z Measurement dimension, laser can measure px, py
   * @param std_laspx Laser measurement noise standard deviation position1 in m
   * @param std_laspy Laser measurement noise standard deviation position2 in m
   * @param Xsig_pred Predicted sigma points matrix
   */
  tuple<VectorXd, MatrixXd> PredictLaserMeasurement(int n_x, int n_aug, int n_z,
                                                    double std_laspx,
                                                    double std_laspy,
                                                    const MatrixXd& Xsig_pred);

  /*
   * Predict Measurement Covariance
   * @param n_x State dimension
   * @param n_aug Augmentation dimension
   * @param n_z Measurement dimension, radar can measure r, phi, and r_dot
   * @param Zsig sigma points in measurement space MatrixXd(n_z, 2 * n_aug + 1)
   * @param Xsig_pred Predicted sigma points matrix
   */
  tuple<VectorXd, MatrixXd> PredictMeasurementCovariance(
      int n_x, int n_aug, int n_z, const MatrixXd& Zsig,
      const MatrixXd& Xsig_pred);

  /*
   * Update State
   * @param n_x State dimension
   * @param n_aug Augmentation dimension
   * @param n_z Measurement dimension, radar can measure r, phi, and r_dot
   * @param is_radar Used to normalised phi
   * @param x State vector
   * @param P State covariance Matrix
   * @param Zsig sigma points in measurement space MatrixXd(n_z, 2 * n_aug + 1)
   * @param z_pred vector for mean predicted measurement
   * @param S predicted measurement covariance
   * @param z vector for incoming measurement
   *
   */
  tuple<VectorXd, MatrixXd> UpdateState(int n_x, int n_aug, int n_z, MeasurementPackage::SensorType sensor_type, 
                                        const MatrixXd& Xsig_pred,
                                        const VectorXd& x, const MatrixXd& P,
                                        const MatrixXd& Zsig,
                                        const VectorXd& z_pred,
                                        const MatrixXd& S, const VectorXd& z);
  /*
   * Sigma Points in Radar Measurement Space
   * @param n_z Measurement dimension, radar can measure r, phi, and r_dot
   * @param n_aug Augmentation dimension
   * @param Xsig_pred Predicted sigma points matrix
   */
  MatrixXd SigmaPointsRadarMeasurementSpace(int n_z, int n_aug,
                                            const MatrixXd& Xsig_pred);

  /*
   * Sigma Points in Laser Measurement Space
   * @param n_z Measurement dimension, lidar can measure px, py
   * @param n_aug Augmentation dimension
   * @param Xsig_pred Predicted sigma points matrix
   */
  MatrixXd SigmaPointsLaserMeasurementSpace(int n_z, int n_aug,
                                            const MatrixXd& Xsig_pred);

  VectorXd WeightsVector(int n_aug, double lambda);
  VectorXd MeanPredictedMeasurement(int n_z, int n_aug, const VectorXd& weights,
                                    const MatrixXd& Zsig);
  MatrixXd MeasurementCovarianceMatrixS(int n_z, int n_aug,
                                        const MatrixXd& Zsig,
                                        const VectorXd& z_pred,
                                        const VectorXd& weights);
  MatrixXd CrossCorrelationMatrix(int n_x, int n_aug, int n_z, 
                                  const MatrixXd& Zsig, const VectorXd& z_pred,
                                  const MatrixXd& Xsig_pred, const VectorXd& x,
                                  const VectorXd& weights);
  MatrixXd NoiseCovarianceMatrixRadar(int n_z, double std_radr,
                                             double std_radphi,
                                             double std_radrd);
  MatrixXd NoiseCovarianceMatrixLaser(int n_z, double std_laspx,
                                             double std_laspy);
  VectorXd InitStateLaserMeasurement(const MeasurementPackage& measurement_pack);
  VectorXd InitStateRadarMeasurement(const MeasurementPackage& measurement_pack);

  // previous timestamp
  long previous_timestamp_;
};

#endif  // UKF_H