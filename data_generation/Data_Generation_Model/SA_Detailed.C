/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2016 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "SA_Detailed.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
tmp<volScalarField> SA_Detailed<BasicTurbulenceModel>::chi() const
{
    return nuTilda_/this->nu();
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SA_Detailed<BasicTurbulenceModel>::fv1
(
    const volScalarField& chi
) const
{
    const volScalarField chi3(pow3(chi));
    return chi3/(chi3 + pow3(Cv1_));
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SA_Detailed<BasicTurbulenceModel>::fv2
(
    const volScalarField& chi,
    const volScalarField& fv1
) const
{
    return 1.0 - chi/(1.0 + chi*fv1);
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SA_Detailed<BasicTurbulenceModel>::Stilda
(
    const volScalarField& chi,
    const volScalarField& fv1
) const
{
    volScalarField Omega(::sqrt(2.0)*mag(skew(fvc::grad(this->U_))));

    return
    (
        max
        (
            Omega
          + fv2(chi, fv1)*nuTilda_/sqr(kappa_*y_),
            Cs_*Omega
        )
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SA_Detailed<BasicTurbulenceModel>::fw
(
    const volScalarField& Stilda
) const
{
    volScalarField r
    (
        min
        (
            nuTilda_
           /(
               max
               (
                   Stilda,
                   dimensionedScalar("SMALL", Stilda.dimensions(), SMALL)
               )
              *sqr(kappa_*y_)
            ),
            scalar(10.0)
        )
    );
    r.boundaryFieldRef() == 0.0;

    const volScalarField g(r + Cw2_*(pow6(r) - r));

    return g*pow((1.0 + pow6(Cw3_))/(pow6(g) + pow6(Cw3_)), 1.0/6.0);
}

template<class BasicTurbulenceModel>
void SA_Detailed<BasicTurbulenceModel>::correctNut
(
    const volScalarField& fv1
)
{
    this->nut_ = nuTilda_*fv1;
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
void SA_Detailed<BasicTurbulenceModel>::correctNut()
{
    correctNut(fv1(this->chi()));
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
SA_Detailed<BasicTurbulenceModel>::SA_Detailed
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    eddyViscosity<RASModel<BasicTurbulenceModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    sigmaNut_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "sigmaNut",
            this->coeffDict_,
            0.66666
        )
    ),
    kappa_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "kappa",
            this->coeffDict_,
            0.41
        )
    ),

    Cb1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cb1",
            this->coeffDict_,
            0.1355
        )
    ),
    Cb2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cb2",
            this->coeffDict_,
            0.622
        )
    ),
    Cw1_(Cb1_/sqr(kappa_) + (1.0 + Cb2_)/sigmaNut_),
    Cw2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cw2",
            this->coeffDict_,
            0.3
        )
    ),
    Cw3_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cw3",
            this->coeffDict_,
            2.0
        )
    ),
    Cv1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cv1",
            this->coeffDict_,
            7.1
        )
    ),
    Cs_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cs",
            this->coeffDict_,
            0.3
        )
    ),

    nuTilda_
    (
        IOobject
        (
            "nuTilda",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),

    y_(wallDist::New(this->mesh_).y()),
    // Filtering operations - simpleFilter
    MyFilter_
    (
        this->mesh_
    )
{
    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool SA_Detailed<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        sigmaNut_.readIfPresent(this->coeffDict());
        kappa_.readIfPresent(this->coeffDict());

        Cb1_.readIfPresent(this->coeffDict());
        Cb2_.readIfPresent(this->coeffDict());
        Cw1_ = Cb1_/sqr(kappa_) + (1.0 + Cb2_)/sigmaNut_;
        Cw2_.readIfPresent(this->coeffDict());
        Cw3_.readIfPresent(this->coeffDict());
        Cv1_.readIfPresent(this->coeffDict());
        Cs_.readIfPresent(this->coeffDict());

        return true;
    }
    else
    {
        return false;
    }
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SA_Detailed<BasicTurbulenceModel>::DnuTildaEff() const
{
    return tmp<volScalarField>
    (
        new volScalarField("DnuTildaEff", (nuTilda_ + this->nu())/sigmaNut_)
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SA_Detailed<BasicTurbulenceModel>::k() const
{
    return tmp<volScalarField>
    (
        new volScalarField
        (
            IOobject
            (
                "k",
                this->runTime_.timeName(),
                this->mesh_
            ),
            this->mesh_,
            dimensionedScalar("0", dimensionSet(0, 2, -2, 0, 0), 0)
        )
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SA_Detailed<BasicTurbulenceModel>::epsilon() const
{
    WarningInFunction
        << "Turbulence kinetic energy dissipation rate not defined for "
        << "Spalart-Allmaras model. Returning zero field"
        << endl;

    return tmp<volScalarField>
    (
        new volScalarField
        (
            IOobject
            (
                "epsilon",
                this->runTime_.timeName(),
                this->mesh_
            ),
            this->mesh_,
            dimensionedScalar("0", dimensionSet(0, 2, -3, 0, 0), 0)
        )
    );
}


template<class BasicTurbulenceModel>
void SA_Detailed<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    const volScalarField chi(this->chi());
    const volScalarField fv1(this->fv1(chi));

    const volScalarField Stilda(this->Stilda(chi, fv1));

    volScalarField nut_prev = this->nut_;

    tmp<fvScalarMatrix> nuTildaEqn
    (
        fvm::ddt(alpha, rho, nuTilda_)
      + fvm::div(alphaRhoPhi, nuTilda_)
      - fvm::laplacian(alpha*rho*DnuTildaEff(), nuTilda_)
      - Cb2_/sigmaNut_*alpha*rho*magSqr(fvc::grad(nuTilda_))
     ==
        Cb1_*alpha*rho*Stilda*nuTilda_
      - fvm::Sp(Cw1_*alpha*rho*fw(Stilda)*nuTilda_/sqr(y_), nuTilda_)
      + fvOptions(alpha, rho, nuTilda_)
    ); // Solved implicitly (therefore fvm)

    nuTildaEqn.ref().relax();
    fvOptions.constrain(nuTildaEqn.ref());
    solve(nuTildaEqn);
    fvOptions.correct(nuTilda_);
    bound(nuTilda_, dimensionedScalar("0", nuTilda_.dimensions(), 0.0));
    nuTilda_.correctBoundaryConditions();

    correctNut(fv1);

    volScalarField nut_diff_ = (this->nut_ - nut_prev);

    if(this->runTime_.outputTime())
    {
        // // Write out SA quantities
        // volScalarField grad_nut_term = Cb2_/sigmaNut_*alpha*rho*magSqr(fvc::grad(nuTilda_));
        // const char* grad_nut_term_name = "grad_nut_term";
        // grad_nut_term.rename(grad_nut_term_name);
        // grad_nut_term.write();

        // volScalarField stilda_term = Cb1_*alpha*rho*Stilda*nuTilda_;
        // const char* stilda_term_name = "stilda_term";
        // stilda_term.rename(stilda_term_name);
        // stilda_term.write();

        // volScalarField source_term = fvc::Sp(Cw1_*alpha*rho*fw(Stilda)*nuTilda_/sqr(y_), nuTilda_);
        // const char* source_term_name = "source_term";
        // source_term.rename(source_term_name);
        // source_term.write();

        // // volScalarField grad_term = magSqr(fvc::grad(nuTilda_));
        // // const char* grad_term_name = "grad_term";
        // // grad_term.rename(grad_term_name);
        // // grad_term.write();

        // // volScalarField lap_term = fvc::laplacian(alpha*rho*DnuTildaEff(), nuTilda_);
        // // const char* lap_term_name = "lap_term";
        // // lap_term.rename(lap_term_name);
        // // lap_term.write();

        // // volScalarField source_term = fvc::Sp(fw(Stilda)*nuTilda_/sqr(y_), nuTilda_);
        // // const char* source_term_name = "source_term";
        // // source_term.rename(source_term_name);
        // // source_term.write();
        
        // // volScalarField stilda_term = Stilda;
        // // const char* stilda_term_name = "stilda_term";
        // // stilda_term.rename(stilda_term_name);
        // // stilda_term.write();

        // // // Velocity related
        // // volVectorField Uf = MyFilter_(this->U_);
        // // const char* Uf_name = "Uf";
        // // Uf.rename(Uf_name);
        // // Uf.write();

        // volScalarField pf_ = this->db().objectRegistry::lookupObject<volScalarField>("p");
        // const char* pf_name = "pf";
        // pf_.rename(pf_name);
        // pf_.write();

        // volScalarField u_ = this->U_.component(vector::X);
        // volScalarField gradu_ = y_*y_*magSqr(fvc::grad(u_));
        // const char* gradu_name = "yygradu";
        // gradu_.rename(gradu_name);
        // gradu_.write();
        
        // volScalarField v_ = this->U_.component(vector::Y);        
        // volScalarField gradv_ = y_*y_*magSqr(fvc::grad(v_));
        // const char* gradv_name = "yygradv";
        // gradv_.rename(gradv_name);
        // gradv_.write();
        
        // volScalarField umag_ = mag(this->U_);
        // const char* umag_name = "umag";
        // umag_.rename(umag_name);
        // umag_.write();

        // // Pressure related
        // const volScalarField& p_ = this->db().objectRegistry::lookupObject<volScalarField>("p");
        // volScalarField gradp_ = y_*y_*magSqr(fvc::grad(p_));
        // const char* gradp_name = "yygradp";
        // gradp_.rename(gradp_name);
        // gradp_.write();

        // // Strain and rotation
        // volScalarField rot_ = sqrt(2.0)*mag(skew(fvc::grad(this->U_)));
        // const char* rot_name = "rot";
        // rot_.rename(rot_name);
        // rot_.write();      

        // volScalarField strain_ = sqrt(2.0)*mag(symm(fvc::grad(this->U_)));
        // const char* strain_name = "strain";
        // strain_.rename(strain_name);
        // strain_.write(); 

        // // Gradients/Laplacian of eddy viscosity
        // volVectorField gradnut_ = fvc::grad(this->nut_); // Magnitude of gradient square
        // const char* gradnut_name = "gradnut";
        // gradnut_.rename(gradnut_name);
        // gradnut_.write(); 
        
        // volScalarField lapnut_ = fvc::laplacian(this->nut_); // Laplacian
        // const char* lapnut_name = "lapnut";
        // lapnut_.rename(lapnut_name);
        // lapnut_.write();

        // // Vorticity
        // volVectorField vort_ = fvc::curl(this->U_);
        // volScalarField vortz_ = vort_.component(vector::Z);
        // const char* vortz_name = "vortz";
        // vortz_.rename(vortz_name);
        // vortz_.write();

        // // Difference prediction
        // const char* nut_diff_name = "nut_diff";
        // nut_diff_.rename(nut_diff_name);
        // nut_diff_.write();

        // Write wall distance
        y_.write();

        // Write the mesh coordinates (just in case needed) - this is written in the Constant
        volScalarField coord_field = this->mesh_.C().component(vector::X);
        const char* cx_name = "cx";
        coord_field.rename(cx_name);
        coord_field.write();

        coord_field = this->mesh_.C().component(vector::Y);
        const char* cy_name = "cy";
        coord_field.rename(cy_name);
        coord_field.write();
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //