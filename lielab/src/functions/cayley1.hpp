#ifndef _LIELAB_FUNCTIONS_CAYLEY1_HPP
#define _LIELAB_FUNCTIONS_CAYLEY1_HPP

namespace lielab
{
    namespace functions
    {
        template <lielab::abstract::LieAlgebra LA>
        lielab::domain::lieiii<LA> cayley1(const LA & g)
        {
            /*! \f{equation*}{ (\mathfrak{g}) \rightarrow G \f}
            *
            * Cayley transform.
            * 
            * \f{equation*}{ cayley1(g) = \frac{Id_g + g/2}{Id_g - g/2} \f}
            * 
            * @param[in] g A Lie algebra.
            * @param[out] G A Lie group.
            * 
            * Source: Engø, Kenth. "On the construction of geometric integrators in the RKMK class."
            * BIT Numerical Mathematics 40.1 (2000): 41-61.
            * 
            * TODO: Restrict this to only work with O, SO, and SP.
            * TODO: Check that this works with SU. Math says no but simulations say otherwise.
            */

            const Eigen::MatrixXd m = g.get_ados_representation();
            const Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(g.shape, g.shape);

            return (Id + m/2.0)*(Id - m/2.0).inverse();
        }

        template <>
        lielab::domain::SU cayley1(const lielab::domain::su & a)
        {
            /*
            * Cayley1 overload for su
            *
            * Needed since su is complex
            */

            const Eigen::MatrixXcd m = a.get_ados_representation();
            const Eigen::MatrixXcd Id = Eigen::MatrixXcd::Identity(a.shape, a.shape);

            return (Id + m/2)*(Id - m/2).inverse();
        }
    }
}

#endif
