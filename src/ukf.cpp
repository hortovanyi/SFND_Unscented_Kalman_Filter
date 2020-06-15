#include "ukf.h"
#include <Eigen/Dense>
#include <iostream>
#include "templates.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Measurement dimensions
  n_laser_z_ = 2;
  n_radar_z_ = 3;

  // State Dimension
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // initial sigma point matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_x_ + 1);

  // augmented state dimension
  n_aug_ = 7;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // std_a_ = 30;
  std_a_ = 5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // std_yawdd_ = 30;
  std_yawdd_ = 5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
}

UKF::~UKF() {}

// State from laser measurements
VectorXd UKF::InitStateLaserMeasurement(
    const MeasurementPackage& measurement_pack) {
  VectorXd x = VectorXd(n_x_);
  x << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0, 0;
  return x;
}

// state from radar measurement
VectorXd UKF::InitStateRadarMeasurement(
    const MeasurementPackage& measurement_pack) {

  VectorXd x = VectorXd(n_x_);

  double rho = measurement_pack.raw_measurements_[0];  // Range - radial distance from origin
  double phi = measurement_pack.raw_measurements_[1];  // Bearing - angle between rho and x
  double rhod = measurement_pack.raw_measurements_[2];  // Radial Velocity - change of p (range rate)

  double px = rho * cos(phi);  // metres
  double py = rho * sin(phi);  // metres
  double v = rhod;  // metres/sec
  double yaw = phi;  // radians
  double yawd = 0.0;  // radians/sec

  x << px, py, v, yaw, yawd;
  return x;
}

void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
    // ignore measurements if we are filtering out radar or laser
  if ((measurement_pack.sensor_type_ == MeasurementPackage::RADAR && !use_radar_)
      || (measurement_pack.sensor_type_ == MeasurementPackage::LASER
          && !use_laser_))
    return;

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    cout << "UKF initialise" << endl;

    // create inital state vector

    switch (measurement_pack.sensor_type_) {
      case MeasurementPackage::RADAR:
        x_ = InitStateRadarMeasurement(measurement_pack);
        break;
      case MeasurementPackage::LASER:
        x_ = InitStateLaserMeasurement(measurement_pack);
        break;
    }

    // initial covariance matrix
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;

   cout << "init x_ = " << x_ << endl;
   cout << "init P_ = " << P_ << endl;
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // calculate delta time and update previous timestamp
  double delta_t = (measurement_pack.timestamp_ - previous_timestamp_)
      / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  /*
   * Prediction
   */
  Prediction(delta_t);

//  cout << "predict x_ = " << x_ << endl;
//  cout << "predict P_ = " << P_ << endl;

  /*
   * Update
   */
  switch (measurement_pack.sensor_type_) {
    case MeasurementPackage::RADAR:
      UpdateRadar(measurement_pack);
      break;
    case MeasurementPackage::LASER:
      UpdateLidar(measurement_pack);
      break;
  }

  // print the output
  // cout << "x_ = " << x_ << endl;
  // cout << "P_ = " << P_ << endl;
}

void UKF::Prediction(double delta_t) {
  MatrixXd Xsig_aug = AugmentedSigmaPoints(n_x_, n_aug_, std_a_, std_yawdd_, x_, P_);
//  cout << "Xsig_aug = " << Xsig_aug << endl;

  Xsig_pred_ = PredictSigmaPoints(n_x_, n_aug_, delta_t, Xsig_aug);
//  cout << "Xsig_pred = " << Xsig_pred_ << endl;

  tie(x_, P_) = PredictMeanAndCovariance(n_x_, n_aug_, Xsig_pred_);
}

/**
 * Generate Sigma Points
 */
MatrixXd UKF::GenerateSigmaPoints(int n_x, const VectorXd& x,
                                  const MatrixXd& P) {
  // calculate spreading parameter
  double lambda = 3 - n_x;

  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x, 2 * n_x + 1);

  //calculate square root of P
  MatrixXd A = P.llt().matrixL();

  //set first column of sigma point matrix
  Xsig.col(0) = x;

  //set remaining sigma points
  for (int i = 0; i < n_x; i++) {
    Xsig.col(i + 1) = x + sqrt(lambda + n_x) * A.col(i);
    Xsig.col(i + 1 + n_x) = x - sqrt(lambda + n_x) * A.col(i);
  }

  return Xsig;
}

/**
 * Augment Sigma Points
 */
MatrixXd UKF::AugmentedSigmaPoints(int n_x, int n_aug, double std_a,
                                   double std_yawdd, const VectorXd& x,
                                   const MatrixXd& P) {
  //define spreading parameter
  double lambda = 3 - n_aug;

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug, n_aug);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

  //create augmented mean state
  x_aug.head(5) = x;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P;
  P_aug(5, 5) = std_a * std_a;
  P_aug(6, 6) = std_yawdd * std_yawdd;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug; i++) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda + n_aug) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug) = x_aug - sqrt(lambda + n_aug) * L.col(i);
  }

  return Xsig_aug;
}

/**
 * Predict Sigma Points
 */
MatrixXd UKF::PredictSigmaPoints(int n_x, int n_aug, double delta_t,
                                 const MatrixXd& Xsig_aug) {
  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  //predict sigma points
  for (int i = 0; i < 2 * n_aug + 1; i++) {
    //extract values for better readability
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0, i) = px_p;
    Xsig_pred(1, i) = py_p;
    Xsig_pred(2, i) = v_p;
    Xsig_pred(3, i) = yaw_p;
    Xsig_pred(4, i) = yawd_p;
  }

  return Xsig_pred;
}

/**
 * Predict Mean and Covariance
 */
tuple<VectorXd, MatrixXd> UKF::PredictMeanAndCovariance(
    int n_x, int n_aug, const MatrixXd& Xsig_pred) {
  //define spreading parameter
  double lambda = 3 - n_aug;

  //create vector for predicted state
  VectorXd x = VectorXd(n_x);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x, n_x);

  //create vector for weights
  VectorXd weights = VectorXd(2 * n_aug + 1);

  // set weights
  double weight_0 = lambda / (lambda + n_aug);
  weights(0) = weight_0;
  for (int i = 1; i < 2 * n_aug + 1; i++) {  //2n+1 weights
    double weight = 0.5 / (n_aug + lambda);
    weights(i) = weight;
  }

  //predicted state mean
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //iterate over sigma points
    x = x + weights(i) * Xsig_pred.col(i);
  }

  //predicted state covariance matrix
  P.fill(0.0);
  for (int i = 1; i < 2 * n_aug + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - Xsig_pred.col(0);

    //angle normalization
    x_diff(3) = normalizeRadiansPiToMinusPi(x_diff(3));

    P = P + weights(i) * x_diff * x_diff.transpose();
  }

  return std::make_tuple(x, P);
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {
   /*
   * Predict Measurement
   */
  // Measurement dimension, laser can measure px, py
  int n_z = n_laser_z_;
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];

  // predict measurement
  VectorXd z_pred;
  MatrixXd S;
  tie(z_pred, S) = PredictLaserMeasurement(n_x_, n_aug_, n_z, std_laspx_,
                                                std_laspy_, Xsig_pred_);

//  cout << "z_pred " << z_pred << endl;
//  cout << "S " << S << endl;

  /*
   * Update State
   */
  MatrixXd Zsig = SigmaPointsLaserMeasurementSpace(n_z, n_aug_, Xsig_pred_);

//  cout << "Zsig " << Zsig << endl;

  tie(x_, P_) = UpdateState(n_x_, n_aug_, n_z, Xsig_pred_, x_, P_, Zsig, z_pred,
                            S, z);
}

MatrixXd UKF::SigmaPointsLaserMeasurementSpace(int n_z, int n_aug,
                                               const MatrixXd& Xsig_pred) {
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug + 1; i++) {
    //2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred(0, i);
    double p_y = Xsig_pred(1, i);

    // measurement model
    Zsig(0, i) = p_x;
    Zsig(1, i) = p_y;

  }
  return Zsig;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
    // Measurement dimension, radar can measure r, phi, and r_dot
  int n_z = n_radar_z_;

  // cout << "UpdateRadar n_z:  "<< n_z << endl;
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package
      .raw_measurements_[2];

  // cout << "z: " << z << endl;

  // predict measurement
  VectorXd z_pred;
  MatrixXd S;
  std::tie(z_pred, S) = PredictRadarMeasurement(n_x_, n_aug_, n_z, std_radr_,
                                                std_radphi_, std_radrd_,
                                                Xsig_pred_);
//  cout << "z_pred " << z_pred << endl;
//  cout << "S " << S << endl;

  MatrixXd Zsig = SigmaPointsRadarMeasurementSpace(n_z, n_aug_, Xsig_pred_);
//  cout << "Zsig " << Zsig << endl;

  tie(x_, P_) = UpdateState(n_x_, n_aug_, n_z, Xsig_pred_, x_, P_, Zsig, z_pred,
                            S, z);

}

MatrixXd UKF::SigmaPointsRadarMeasurementSpace(int n_z, int n_aug,
                                               const MatrixXd& Xsig_pred) {
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug + 1; i++) {
    //2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred(0, i);
    double p_y = Xsig_pred(1, i);
    double v = Xsig_pred(2, i);
    double yaw = Xsig_pred(3, i);
    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;
    // measurement model
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);  //r
    Zsig(1, i) = atan2(p_y, p_x);  //phi
    if (Zsig(0, i) != 0) {
      Zsig(2, i) = (p_x * v1 + p_y * v2) / Zsig(0, i);  //r_dot
    } else {
      Zsig(2, i) = 0;
    }

  }
  return Zsig;
}

MatrixXd UKF::NoiseCovarianceMatrixRadar(int n_z, double std_radr,
                                         double std_radphi, double std_radrd) {
  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr * std_radr, 0, 0, 0, std_radphi * std_radphi, 0, 0, 0, std_radrd
      * std_radrd;
  return R;
}

/*
 * Predict Radar Measurement
 */
tuple<VectorXd, MatrixXd> UKF::PredictRadarMeasurement(
    int n_x, int n_aug, int n_z, double std_radr, double std_radphi,
    double std_radrd, const MatrixXd& Xsig_pred) {

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = SigmaPointsRadarMeasurementSpace(n_z, n_aug, Xsig_pred);

  MatrixXd S;
  VectorXd z_pred;
  tie(z_pred, S) = PredictMeasurementCovariance(n_x, n_aug, n_z, Zsig,
                                                Xsig_pred);

  //add measurement noise covariance matrix
  MatrixXd R = NoiseCovarianceMatrixRadar(n_z, std_radr, std_radphi, std_radrd);
  S = S + R;

  return std::make_tuple(z_pred, S);
}

MatrixXd UKF::NoiseCovarianceMatrixLaser(int n_z, double std_laspx,
                                         double std_laspy) {
  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx * std_laspx, 0, 0, std_laspy * std_laspy;
  return R;
}

/*
 * Predict Laser Measurement
 */
tuple<VectorXd, MatrixXd> UKF::PredictLaserMeasurement(
    int n_x, int n_aug, int n_z, double std_laspx, double std_laspy,
    const MatrixXd& Xsig_pred) {

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = SigmaPointsLaserMeasurementSpace(n_z, n_aug, Xsig_pred);

  MatrixXd S;
  VectorXd z_pred;
  tie(z_pred, S) = PredictMeasurementCovariance(n_x, n_aug, n_z, Zsig,
                                                Xsig_pred);

  //add measurement noise covariance matrix
  MatrixXd R = NoiseCovarianceMatrixLaser(n_z, std_laspx, std_laspy);
  S = S + R;

  return std::make_tuple(z_pred, S);
}

VectorXd UKF::MeanPredictedMeasurement(int n_z, int n_aug,
                                       const VectorXd& weights,
                                       const MatrixXd& Zsig) {
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {
    z_pred = z_pred + weights(i) * Zsig.col(i);
  }
  return z_pred;
}

MatrixXd UKF::MeasurementCovarianceMatrixS(int n_z, int n_aug,
                                           const MatrixXd& Zsig,
                                           const VectorXd& z_pred,
                                           const VectorXd& weights) {
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
//  for (int i = 0; i < 2 * n_aug + 1; i++) {
  for (int i = 1; i < 2 * n_aug + 1; i++) {
    //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - Zsig.col(0);

    //angle normalization
    z_diff(1) = normalizeRadiansPiToMinusPi(z_diff(1));
    S = S + weights(i) * z_diff * z_diff.transpose();
  }
  return S;
}

/*
 * Predict Laser Measurement
 */
tuple<VectorXd, MatrixXd> UKF::PredictMeasurementCovariance(
    int n_x, int n_aug, int n_z, const MatrixXd& Zsig,
    const MatrixXd& Xsig_pred) {

  //define spreading parameter
  double lambda = 3 - n_aug;

  //set vector for weights
  VectorXd weights = WeightsVector(n_aug, lambda);

  //mean predicted measurement
  VectorXd z_pred = MeanPredictedMeasurement(n_z, n_aug, weights, Zsig);

  //measurement covariance matrix S
  MatrixXd S = MeasurementCovarianceMatrixS(n_z, n_aug, Zsig, z_pred, weights);

  return make_tuple(z_pred, S);

}

VectorXd UKF::WeightsVector(int n_aug, double lambda) {
  //set vector for weights
  VectorXd weights = VectorXd(2 * n_aug + 1);
  double weight_0 = lambda / (lambda + n_aug);
  weights(0) = weight_0;
  for (int i = 1; i < 2 * n_aug + 1; i++) {
    //2n+1 weights
    double weight = 0.5 / (n_aug + lambda);
    weights(i) = weight;
  }

  return weights;
}

MatrixXd UKF::CrossCorrelationMatrix(int n_x, int n_aug, int n_z,
                                     const MatrixXd& Zsig,
                                     const VectorXd& z_pred,
                                     const MatrixXd& Xsig_pred,
                                     const VectorXd& x,
                                     const VectorXd& weights) {

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
//  for (int i = 0; i < 2 * n_aug + 1; i++) {
  for (int i = 1; i < 2 * n_aug + 1; i++) {
    //2n+1 sigma points
    //residual
//    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd z_diff = Zsig.col(i) - Zsig.col(0);
    //angle normalization
    z_diff(1) = normalizeRadiansPiToMinusPi(z_diff(1));
//    VectorXd x_diff = Xsig_pred.col(i) - x;
    VectorXd x_diff = Xsig_pred.col(i) - Xsig_pred.col(0);
    //angle normalization
    x_diff(3) = normalizeRadiansPiToMinusPi(x_diff(3));
    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }

  return Tc;
}

/*
 * Update State
 */
tuple<VectorXd, MatrixXd> UKF::UpdateState(int n_x, int n_aug, int n_z,
                                           const MatrixXd& Xsig_pred,
                                           const VectorXd& x, const MatrixXd& P,
                                           const MatrixXd& Zsig,
                                           const VectorXd& z_pred,
                                           const MatrixXd& S,
                                           const VectorXd& z) {

  //define spreading parameter
  double lambda = 4 - n_aug;

  //set vector for weights
  VectorXd weights = WeightsVector(n_aug, lambda);

  //calculate cross correlation matrix
  MatrixXd Tc = CrossCorrelationMatrix(n_x, n_aug, n_z, Zsig, z_pred, Xsig_pred,
                                       x, weights);

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  z_diff(1) = normalizeRadiansPiToMinusPi(z_diff(1));

  //update state mean and covariance matrix
  VectorXd x_update = x + K * z_diff;
  MatrixXd P_update = P - K * S * K.transpose();

  return make_tuple(x_update, P_update);
}