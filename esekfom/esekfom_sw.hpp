/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Author: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

/*
 *  This file is modified by:
 *  Eungchang Mason Lee (eungchang_mason@kaist.ac.kr)
 */

#ifndef ESEKFOM_EKF_HPP
#define ESEKFOM_EKF_HPP


#include <vector>
#include <cstdlib>

#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "../mtk/types/vect.hpp"
#include "../mtk/types/SOn.hpp"
#include "../mtk/types/S2.hpp"
#include "../mtk/startIdx.hpp"
#include "../mtk/build_manifold.hpp"
/// TBB
#include <tbb/tbb.h> // For parallel programming


namespace esekfom
{
    // used for iterated error state EKF update
    // for the aim to calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise covariance (R) at the same time, by only one function.
    // applied for measurement as an Eigen matrix whose dimension is changing
    template<typename T>
    struct dyn_share_datastruct
    {
        bool valid;
        bool converge;
        Eigen::Matrix<T, Eigen::Dynamic, 1> z;
        Eigen::Matrix<T, Eigen::Dynamic, 1> h;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_v;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> R;
    };

    template<typename state, int process_noise_dof, typename input = state, typename measurement = state, int measurement_noise_dof = 0>
    class esekf
    {
        typedef esekf self;
        enum
        {
            n = state::DOF,
            m = state::DIM,
            l = measurement::DOF
        };

    public:
        typedef typename state::scalar scalar_type;
        typedef Eigen::Matrix<scalar_type, n, n> cov;
        typedef Eigen::Matrix<scalar_type, m, n> cov_;
        typedef Eigen::SparseMatrix<scalar_type> spMt;
        typedef Eigen::Matrix<scalar_type, n, 1> vectorized_state;
        typedef Eigen::Matrix<scalar_type, m, 1> flatted_state;
        typedef flatted_state processModel(state &, const input &);
        typedef Eigen::Matrix<scalar_type, m, n> processMatrix1(state &, const input &);
        typedef Eigen::Matrix<scalar_type, m, process_noise_dof> processMatrix2(state &, const input &);
        typedef Eigen::Matrix<scalar_type, process_noise_dof, process_noise_dof> processnoisecovariance;
        typedef void measurementModel_dyn_share(state &, dyn_share_datastruct<scalar_type> &);
        typedef Eigen::Matrix<scalar_type, l, n> measurementMatrix1(state &, bool &);
        typedef Eigen::Matrix<scalar_type, l, measurement_noise_dof> measurementMatrix2(state &, bool &);

        esekf(const state &x = state(),
              const cov &P = cov::Identity()):
            x_(x), P_(P){};

        // receive system-specific models and their differentions
        // for measurement as an Eigen matrix whose dimension is changing.
        // calculate  measurement (z), estimate measurement (h), partial differention matrices (h_x, h_v) and the noise covariance (R) at the same time, by only one function (h_dyn_share_in).
        void init_dyn_share(processModel f_in,
                            processMatrix1 f_x_in,
                            processMatrix2 f_w_in,
                            measurementModel_dyn_share h_dyn_share_in,
                            int maximum_iteration,
                            scalar_type limit_vector[n])
        {
            f = f_in;
            f_x = f_x_in;
            f_w = f_w_in;
            h_dyn_share = h_dyn_share_in;

            maximum_iter = maximum_iteration;
            for (int i = 0; i < n; i++)
            {
                limit[i] = limit_vector[i];
            }

            x_.build_S2_state();
            x_.build_SO3_state();
            x_.build_vect_state();
        }
        void init_dyn_share(processModel f_in,
                            processMatrix1 f_x_in,
                            processMatrix2 f_w_in,
                            std::function<void(state &, dyn_share_datastruct<double> &)> h_dyn_share_in,
                            int maximum_iteration,
                            scalar_type limit_vector[n])
        {
            f = f_in;
            f_x = f_x_in;
            f_w = f_w_in;
            h_dyn_share_function = h_dyn_share_in;

            maximum_iter = maximum_iteration;
            for (int i = 0; i < n; i++)
            {
                limit[i] = limit_vector[i];
            }

            x_.build_S2_state();
            x_.build_SO3_state();
            x_.build_vect_state();

            // note the state structure
            // m: 18 + 6*SW (all states), n: 17 + 6*SW (all states - 1, Gravity as S2)
            // process_noise_dof: 12 (IMU 6 + IMU bias 6)
            // x_.vect_state.size(): 5 + SW size
            // for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++) {
            // int idx = (*it).first.first; // index in n
            // int dim = (*it).first.second; // index in m
            // int dof = (*it).second; // 3 fixed, 3D vector
            // }
            // x_.SO3_state.size(): 2 + SW size
            // for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
            // 	int idx = (*it).first; // index in n
            // 	int dim = (*it).second; // index in m
            // }
            // x_.S2_state.size(): 1 fixed
            // for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
            // int idx = (*it).first; // index in n, 21 fixed
            // int dim = (*it).second; // index in m, 21 fixed
            // }
        }

        // iterated error state EKF propogation
        void predict(double &_dt,
                     processnoisecovariance &_Q,
                     const input &_i_in)
        {
            flatted_state f_ = f(x_, _i_in);
            cov_ f_x_ = f_x(x_, _i_in);
            cov f_x_final;

            Eigen::Matrix<scalar_type, m, process_noise_dof> f_w_ = f_w(x_, _i_in);
            Eigen::Matrix<scalar_type, n, process_noise_dof> f_w_final;
            state x_before = x_;
            x_.oplus(f_, _dt);

            F_x1 = cov::Identity();
            // clang-format off
            tbb::parallel_for<size_t>(0, x_.vect_state.size(), 1, [&](size_t vect_idx_)
            {
                int idx = x_.vect_state[vect_idx_].first.first;
                int dim = x_.vect_state[vect_idx_].first.second;
                int dof = x_.vect_state[vect_idx_].second;
                for(int i = 0; i < n; i++)
                {
                    for(int j=0; j<dof; j++)
                    {
                        f_x_final(idx+j, i) = f_x_(dim+j, i);
                    }
                }
                for(int i = 0; i < process_noise_dof; i++)
                {
                    for(int j=0; j<dof; j++)
                    {
                        f_w_final(idx+j, i) = f_w_(dim+j, i);
                    }
                }
            });
            tbb::parallel_for<size_t>(0, x_.SO3_state.size(), 1, [&](size_t so3_idx_)
            {
                Eigen::Matrix<scalar_type, 3, 3> res_temp_SO3;
                MTK::vect<3, scalar_type> seg_SO3;
                int idx = x_.SO3_state[so3_idx_].first;
                int dim = x_.SO3_state[so3_idx_].second;
                for(int i = 0; i < 3; i++)
                {
                    seg_SO3(i) = -1 * f_(dim + i) * _dt;
                }
                MTK::SO3<scalar_type> res;
                res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, 0.5); //note
                // res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));

                F_x1.template block<3, 3>(idx, idx) = res.toRotationMatrix();
                res_temp_SO3 = MTK::A_matrix(seg_SO3);
                for(int i = 0; i < n; i++)
                {
                    f_x_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_. template block<3, 1>(dim, i));	
                }
                for(int i = 0; i < process_noise_dof; i++)
                {
                    f_w_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_. template block<3, 1>(dim, i));
                }
            });
            // clang-format on

            Eigen::Matrix<scalar_type, 2, 3> res_temp_S2;
            Eigen::Matrix<scalar_type, 2, 2> res_temp_S2_;
            MTK::vect<3, scalar_type> seg_S2;
            for (std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
            {
                int idx = (*it).first;
                int dim = (*it).second;
                for (int i = 0; i < 3; i++)
                {
                    seg_S2(i) = f_(dim + i) * _dt;
                }
                MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
                MTK::SO3<scalar_type> res;
                res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_S2, 0.5); //note
                // res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_S2, scalar_type(1 / 2));
                Eigen::Matrix<scalar_type, 2, 3> Nx;
                Eigen::Matrix<scalar_type, 3, 2> Mx;
                x_.S2_Nx_yy(Nx, idx);
                x_before.S2_Mx(Mx, vec, idx);

                F_x1.template block<2, 2>(idx, idx) = Nx * res.toRotationMatrix() * Mx;
                Eigen::Matrix<scalar_type, 3, 3> x_before_hat;
                x_before.S2_hat(x_before_hat, idx);
                res_temp_S2 = -Nx * res.toRotationMatrix() * x_before_hat * MTK::A_matrix(seg_S2).transpose();

                for (int i = 0; i < n; i++)
                {
                    f_x_final.template block<2, 1>(idx, i) = res_temp_S2 * (f_x_.template block<3, 1>(dim, i));
                }
                for (int i = 0; i < process_noise_dof; i++)
                {
                    f_w_final.template block<2, 1>(idx, i) = res_temp_S2 * (f_w_.template block<3, 1>(dim, i));
                }
            }

            F_x1 += f_x_final * _dt;
            P_ = (F_x1)*P_ * (F_x1).transpose() + (_dt * f_w_final) * _Q * (_dt * f_w_final).transpose();
        }

        // iterated error state EKF update modified for one specific system.
        template<int SLIDING_WINDOW_SIZE, int PASSIVE_WINDOW_SIZE>
        void update_iterated_dyn_share_modified(double _R,
                                                bool _bound_function = false)
        {
            dyn_share_datastruct<scalar_type> dyn_share;
            dyn_share.valid = true;
            dyn_share.converge = true;
            int t = 0;
            state x_propagated = x_;
            cov P_propagated = P_;
            int dof_Measurement;

            Eigen::Matrix<scalar_type, n, 1> K_h;
            Eigen::Matrix<scalar_type, n, n> K_x;
            Eigen::Matrix<scalar_type, n, n> K_R_KT;
            Eigen::Matrix<scalar_type, n, n> K_R_KT_active = Eigen::Matrix<scalar_type, n, n>::Zero();

            vectorized_state dx_new = vectorized_state::Zero();
            for (int i = -1; i < maximum_iter; i++)
            {
                dyn_share.valid = true;
                if (_bound_function)
                {
                    h_dyn_share_function(x_, dyn_share);
                }
                else
                {
                    h_dyn_share(x_, dyn_share);
                }

                if (!dyn_share.valid)
                {
                    continue;
                }

                // idea: UPDATE_JACOBIAN_SIZE is for fast calculation, FIXED_STATE_SIZE is for fixed states (K = 0)
                constexpr int UPDATE_JACOBIAN_SIZE = 6 + SLIDING_WINDOW_SIZE * 6;
                constexpr int FIXED_STATE_SIZE = PASSIVE_WINDOW_SIZE * 6;
                // Eigen::Matrix<scalar_type, Eigen::Dynamic, 6> h_x_ = dyn_share.h_x;
                Eigen::Matrix<scalar_type, Eigen::Dynamic, UPDATE_JACOBIAN_SIZE> h_x_ = dyn_share.h_x; // note the dimension: (1+sw size) x 6
                dof_Measurement = h_x_.rows();
                vectorized_state dx;
                x_.boxminus(dx, x_propagated);
                dx_new = dx;
                P_ = P_propagated;

                // clang-format off
                tbb::parallel_for<size_t>(0, x_.SO3_state.size(), 1, [&](size_t so3_idx_)
                {
                    Eigen::Matrix<scalar_type, 3, 3> res_temp_SO3;
                    MTK::vect<3, scalar_type> seg_SO3;
                    int idx = x_.SO3_state[so3_idx_].first;
                    for(int i = 0; i < 3; i++)
                    {
                        seg_SO3(i) = dx(idx+i);
                    }
                    res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
                    dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
                    for(int i = 0; i < n; i++)
                    {
                        P_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i));	
                    }
                    for(int i = 0; i < n; i++)
                    {
                        P_. template block<1, 3>(i, idx) =(P_. template block<1, 3>(i, idx)) *  res_temp_SO3.transpose();	
                    }
                });
                // clang-format on

                Eigen::Matrix<scalar_type, 2, 2> res_temp_S2;
                MTK::vect<2, scalar_type> seg_S2;
                for (std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
                {
                    int idx = (*it).first;
                    // int dim = (*it).second;
                    for (int i = 0; i < 2; i++)
                    {
                        seg_S2(i) = dx(idx + i);
                    }
                    Eigen::Matrix<scalar_type, 2, 3> Nx;
                    Eigen::Matrix<scalar_type, 3, 2> Mx;
                    x_.S2_Nx_yy(Nx, idx);
                    x_propagated.S2_Mx(Mx, seg_S2, idx);
                    res_temp_S2 = Nx * Mx;
                    dx_new.template block<2, 1>(idx, 0) = res_temp_S2 * dx_new.template block<2, 1>(idx, 0);
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
                    }
                    for (int i = 0; i < n; i++)
                    {
                        P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                    }
                }

                if (n > dof_Measurement)
                {
                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x_cur = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, n);
                    // h_x_cur.topLeftCorner(dof_Measurement, 6) = h_x_;
                    h_x_cur.topRightCorner(dof_Measurement, UPDATE_JACOBIAN_SIZE) = h_x_; //note

                    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_ = P_ * h_x_cur.transpose() * (h_x_cur * P_ * h_x_cur.transpose() / _R + Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(dof_Measurement, dof_Measurement)).inverse() / _R;
                    K_h = K_ * dyn_share.h;
                    K_x = K_ * h_x_cur;
                    K_R_KT = K_ * K_.transpose() * _R; //note
                }
                else
                {
                    cov P_temp = (P_ / _R).inverse();
                    // Eigen::Matrix<scalar_type, 6, 6> HTH = h_x_.transpose() * h_x_;
                    // P_temp.template block<6, 6>(0, 0) += HTH;
                    Eigen::Matrix<scalar_type, UPDATE_JACOBIAN_SIZE, UPDATE_JACOBIAN_SIZE> HTH = h_x_.transpose() * h_x_;                         //note
                    P_temp.template block<UPDATE_JACOBIAN_SIZE, UPDATE_JACOBIAN_SIZE>(n - UPDATE_JACOBIAN_SIZE, n - UPDATE_JACOBIAN_SIZE) += HTH; //note
                    cov P_inv = P_temp.inverse();

                    // K_h = P_inv.template block<n, 6>(0, 0) * h_x_.transpose() * dyn_share.h;
                    K_h = P_inv.template block<n, UPDATE_JACOBIAN_SIZE>(0, n - UPDATE_JACOBIAN_SIZE) * h_x_.transpose() * dyn_share.h; //note

                    K_x.setZero();
                    // K_x.template block<n, 6>(0, 0) = P_inv.template block<n, 6>(0, 0) * HTH;
                    K_x.template block<n, UPDATE_JACOBIAN_SIZE>(0, n - UPDATE_JACOBIAN_SIZE) = P_inv.template block<n, UPDATE_JACOBIAN_SIZE>(0, n - UPDATE_JACOBIAN_SIZE) * HTH;                             //note
                    K_R_KT = P_inv.template block<n, UPDATE_JACOBIAN_SIZE>(0, n - UPDATE_JACOBIAN_SIZE) * HTH * P_inv.template block<n, UPDATE_JACOBIAN_SIZE>(0, n - UPDATE_JACOBIAN_SIZE).transpose() * _R; //note Joseph form
                }
                K_R_KT_active.template block<n - FIXED_STATE_SIZE, n - FIXED_STATE_SIZE>(0, 0) = K_R_KT.template block<n - FIXED_STATE_SIZE, n - FIXED_STATE_SIZE>(0, 0); //note
                K_h.template block<FIXED_STATE_SIZE, 1>(n - FIXED_STATE_SIZE, 0).setZero();                                                                               //note
                K_x.template block<FIXED_STATE_SIZE, n>(n - FIXED_STATE_SIZE, 0).setZero();                                                                               //note
                Eigen::Matrix<scalar_type, n, n> I_with_zeros = Eigen::Matrix<scalar_type, n, n>::Identity();                                                             //note
                I_with_zeros.template block<FIXED_STATE_SIZE, FIXED_STATE_SIZE>(n - FIXED_STATE_SIZE, n - FIXED_STATE_SIZE).setZero();                                    //note

                Eigen::Matrix<scalar_type, n, 1> dx_ = K_h + (K_x - I_with_zeros) * dx_new; //note
                x_.boxplus(dx_);
                dyn_share.converge = true;
                for (int i = 0; i < n; i++)
                {
                    if (std::fabs(dx_[i]) > limit[i])
                    {
                        dyn_share.converge = false;
                        break;
                    }
                }
                if (dyn_share.converge)
                {
                    t++;
                }
                if (!t && i == maximum_iter - 2)
                {
                    dyn_share.converge = true;
                }

                if (t > 1 || i == maximum_iter - 1)
                {
                    // L_ = P_; //todo delete

                    // clang-format off
                    tbb::parallel_for<size_t>(0, x_.SO3_state.size(), 1, [&](size_t so3_idx_)
                    {
                        Eigen::Matrix<scalar_type, 3, 3> res_temp_SO3;
                        MTK::vect<3, scalar_type> seg_SO3;
                        int idx = x_.SO3_state[so3_idx_].first;
                        for(int i = 0; i < 3; i++)
                        {
                            seg_SO3(i) = dx_(i + idx);
                        }
                        res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
                        // for(int i = 0; i < n; i++)
                        // {
                            // L_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i)); //todo delete
                        // }
                        // for(int i = 0; i < 6; i++)
                        for(int i = n-UPDATE_JACOBIAN_SIZE; i < n-FIXED_STATE_SIZE; i++) //note
                        {
                            K_x. template block<3, 1>(idx, i) = res_temp_SO3 * (K_x. template block<3, 1>(idx, i));
                        }
                        for(int i = 0; i < n; i++)
                        {
                            // L_. template block<1, 3>(i, idx) = (L_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose(); //todo delete
                            P_. template block<1, 3>(i, idx) = (P_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
                        }
                    });
                    // clang-format on

                    Eigen::Matrix<scalar_type, 2, 2> res_temp_S2;
                    MTK::vect<2, scalar_type> seg_S2;
                    for (typename std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
                    {
                        int idx = (*it).first;
                        for (int i = 0; i < 2; i++)
                        {
                            seg_S2(i) = dx_(i + idx);
                        }

                        Eigen::Matrix<scalar_type, 2, 3> Nx;
                        Eigen::Matrix<scalar_type, 3, 2> Mx;
                        x_.S2_Nx_yy(Nx, idx);
                        x_propagated.S2_Mx(Mx, seg_S2, idx);
                        res_temp_S2 = Nx * Mx;
                        // for (int i = 0; i < n; i++)
                        // {
                        // L_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i)); //todo delete
                        // }
                        // for (int i = n-UPDATE_JACOBIAN_SIZE; i < n; i++) //note
                        for (int i = n - UPDATE_JACOBIAN_SIZE; i < n - FIXED_STATE_SIZE; i++) //note
                        {
                            K_x.template block<2, 1>(idx, i) = res_temp_S2 * (K_x.template block<2, 1>(idx, i));
                        }
                        for (int i = 0; i < n; i++)
                        {
                            // L_.template block<1, 2>(i, idx) = (L_.template block<1, 2>(i, idx)) * res_temp_S2.transpose(); //todo delete
                            P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
                        }
                    }

                    // P_ = L_ - K_x.template block<n, 6>(0, 0) * P_.template block<6, n>(0, 0);
                    // idea: Joseph form is essential for numerical stability
                    Eigen::Matrix<scalar_type, n, n> I_KH = Eigen::Matrix<scalar_type, n, n>::Identity() - K_x;
                    P_ = I_KH * P_ * I_KH.transpose() + K_R_KT;
                    return;
                }
            }
        }

        void change_x(state &_input_state)
        {
            x_ = _input_state;
            if ((!x_.vect_state.size()) && (!x_.SO3_state.size()) && (!x_.S2_state.size()))
            {
                x_.build_S2_state();
                x_.build_SO3_state();
                x_.build_vect_state();
            }
            return;
        }
        void change_P(cov &_input_cov)
        {
            P_ = _input_cov;
            return;
        }
        const state &get_x() const
        {
            return x_;
        }
        const cov &get_P() const
        {
            return P_;
        }

    private:
        state x_;
        measurement m_;
        cov P_;
        cov F_x1 = cov::Identity();
        cov L_ = cov::Identity();

        processModel *f;
        processMatrix1 *f_x;
        processMatrix2 *f_w;

        measurementModel_dyn_share *h_dyn_share;
        measurementMatrix1 *h_x;
        measurementMatrix2 *h_v;

        std::function<void(state &, dyn_share_datastruct<double> &)> h_dyn_share_function;

        int maximum_iter = 0;
        scalar_type limit[n];

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP
